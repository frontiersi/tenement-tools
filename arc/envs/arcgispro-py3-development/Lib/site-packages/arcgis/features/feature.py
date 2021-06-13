"""
In the GIS, entities located in space with a set of properties can be represented as features. This module has the types
to represent features and collection of features.
"""
import copy
import json
import ujson as _ujson
import os
import re
import tempfile
import uuid
from datetime import datetime
from arcgis._impl.common._mixins import PropertyMap
from arcgis._impl.common._spatial import json_to_featureclass
from arcgis._impl.common._utils import _date_handler
from arcgis.geometry import BaseGeometry, Point, MultiPoint, Polyline, Polygon, Geometry, SpatialReference
from arcgis.gis import Layer

try:
    import arcpy
    HASARCPY = True
except:
    HASARCPY = False

class Feature(object):
    """ Entities located in space with a set of properties can be represented as features.

    .. code-block:: python

        # Obtain a feature from a feature layer:

        feat_set = feature_layer.query(where="OBJECTID=1")
        feat = feat_set[0]

    """
    _geom = None
    _json = None
    _dict = None
    _geom_type = None
    _attributes = None
    _wkid = None

    # ----------------------------------------------------------------------
    def __init__(self, geometry=None, attributes=None):
        """Constructor"""
        self._dict = {}
        if geometry is not None:
            self._dict["geometry"] = geometry
        if attributes is not None:
            self._dict["attributes"] = attributes

    # ----------------------------------------------------------------------
    def set_value(self, field_name, value):
        """
        Sets an attribute value for a given field name

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        field_name          Required String. The name of the field to update.
        ---------------     --------------------------------------------------------------------
        value               Required. Value to update the field with.
        ===============     ====================================================================

        :return: boolean indicating whether field_name value was updated.

        """
        if field_name in self.fields:
            if value is not None:
                self._dict['attributes'][field_name] = value
                self._json = json.dumps(self._dict, default=_date_handler)
            else:
                pass
        elif field_name.upper() in ['SHAPE', 'SHAPE@', "GEOMETRY"]:
            if isinstance(value, BaseGeometry):
                if isinstance(value, Point):
                    self._dict['geometry'] = {
                        "x": value['x'],
                        "y": value['y']
                    }
                elif isinstance(value, MultiPoint):
                    self._dict['geometry'] = {
                        "points": value['points']
                    }
                elif isinstance(value, Polyline):
                    self._dict['geometry'] = {
                        "paths": value['paths']
                    }
                elif isinstance(value, Polygon):
                    self._dict['geometry'] = {
                        "rings": value['rings']
                    }
                else:
                    return False
                self._json = json.dumps(self._dict, default=_date_handler)
        else:
            return False
        return True

    # ----------------------------------------------------------------------
    def get_value(self, field_name):
        """
        Retrieves the value for a specified field name

        +--------------+----+------------------------------------------------------------------+
        |**Argument**  |    |**Description**                                                   |
        +==============+====+==================================================================+
        | field name   |    | | Required String. The name of the field to get the value for.   |
        |              |    | |                                                                |
        |              |    | | ``feature.fields`` will return a list of all field names.      |
        +--------------+----+------------------------------------------------------------------+

        :return: value for the specified attribute field of the feature.

        """
        if field_name in self.fields:
            return self._dict['attributes'][field_name]
        elif field_name is not None and field_name.upper() in ['SHAPE', 'SHAPE@', "GEOMETRY"]:
            return self._dict['geometry']
        return None

    # ----------------------------------------------------------------------
    @property
    def as_dict(self):
        """:return: the feature as a dictionary"""
        return self._dict

    # ----------------------------------------------------------------------
    @property
    def as_row(self):
        """ :return: the feature as a tuple containing two lists:

        =============     ===========================================================
        **List of:**          **Description**
        -------------     -----------------------------------------------------------
        row values        the specific attribute values and geometry for this feature
        -------------     -----------------------------------------------------------
        field names       the name for each attribute field
        =============     ===========================================================
        """
        fields = self.fields
        row = [""] * len(fields)
        for key, val in self._attributes.items():
            row[fields.index(key)] = val
            del val
            del key
        if self.geometry is not None:
            row.append(self.geometry)
            fields.append("SHAPE@")
        return row, fields

    # ----------------------------------------------------------------------
    @property
    def geometry(self):
        """ :return: the feature geometry"""
        if self._geom is None:
            if 'geometry' in self._dict.keys():
                self._geom = self._dict['geometry']
            else:
                return None
        return self._geom

    @geometry.setter
    def geometry(self, value):
        """gets/sets a feature's geometry"""
        self._geom = value
        self._dict['geometry'] = value

    # ----------------------------------------------------------------------
    @property
    def attributes(self):
        """:return: a dictionary of feature attribute values with field names as the key"""
        if self._attributes is None and 'attributes' in self._dict:
            self._attributes = self._dict['attributes']
        return self._attributes

    @attributes.setter
    def attributes(self, value):
        """gets/sets a feature's attributes"""
        self._attributes = value
        self._dict['attributes'] = value

    # ----------------------------------------------------------------------
    @property
    def fields(self):
        """ :return: attribute field names for the feature as a list of strings"""
        if 'attributes' in self._dict:
            self._attributes = self._dict['attributes']
            return list(self._attributes.keys())
        else:
            return []

    # ----------------------------------------------------------------------
    @property
    def geometry_type(self):
        """ :return: the geometry type of the feature as a string"""
        if self._geom_type is None:
            if self.geometry is not None:
                if hasattr(self.geometry, 'type'):
                    self._geom_type = self.geometry.type
                else:
                    self._geom_type = Geometry(self.geometry)._type
            else:
                self._geom_type = "Table"
        return self._geom_type

    # ----------------------------------------------------------------------
    @classmethod
    def from_json(cls, json_str):
        """:return: a feature from a JSON string"""
        feature = _ujson.loads(json_str)
        geom = feature['geometry'] if 'geometry' in feature else None
        attribs = feature['attributes'] if 'attributes' in feature else None
        return cls(geom, attribs)

    # ----------------------------------------------------------------------
    @classmethod
    def from_dict(cls, feature, sr=None):
        """:return: a feature from a dict"""
        geom = feature['geometry'] if 'geometry' in feature else None
        if geom and \
           sr and \
           isinstance(geom, dict) and \
           not 'spatialReference' in geom:
            geom['spatialReference'] = sr

        attribs = feature['attributes'] if 'attributes' in feature else None
        if 'centroid' in feature:
            if attribs is None:
                attribs = {'centroid' : feature['centroid']}
            elif 'centroid' in attribs:
                fld = "centroid_" + uuid.uuid4().hex[:2]
                attribs[fld] = feature['centroid']
            else:
                attribs['centroid'] = feature['centroid']
        return cls(geom, attribs)
    # ----------------------------------------------------------------------
    def __str__(self):
        """"""
        return json.dumps(self.as_dict, default=_date_handler)

    __repr__ = __str__


class FeatureSet(object):
    """
    A set of features with information about their fields, field aliases, geometry type, spatial reference etc.

    FeatureSets are commonly used as input/output with several Geoprocessing
    Tools, and can be the obtained through the query() methods of feature layers.
    A FeatureSet can be combined with a layer definition to compose a FeatureCollection.

    FeatureSet contains Feature objects, including the values for the
    fields requested by the user. For layers, if you request geometry
    information, the geometry of each feature is also returned in the
    FeatureSet. For tables, the FeatureSet does not include geometries.

    If a Spatial Reference is not specified at the FeatureSet level, the
    FeatureSet will assume the SpatialReference of its first feature. If
    the SpatialReference of the first feature is also not specified, the
    spatial reference will be UnknownCoordinateSystem.
    """
    _fields = None
    _features = None
    _has_z = None
    _has_m = None
    _geometry_type = None
    _spatial_reference = None
    _object_id_field_name = None
    _global_id_field_name = None
    _display_field_name = None
    _allowed_geom_types = ["esriGeometryPoint", "esriGeometryMultipoint", "esriGeometryPolyline",
                           "esriGeometryPolygon", "esriGeometryEnvelope"]

    # ----------------------------------------------------------------------
    def __init__(self,
                 features,
                 fields=None,
                 has_z=False,
                 has_m=False,
                 geometry_type=None,
                 spatial_reference=None,
                 display_field_name=None,
                 object_id_field_name=None,
                 global_id_field_name=None):
        """Constructor"""
        self._fields = fields

        self._has_z = has_z
        self._has_m = has_m
        self._geometry_type = geometry_type
        self._spatial_reference = spatial_reference
        self._display_field_name = display_field_name
        self._object_id_field_name = object_id_field_name
        self._global_id_field_name = global_id_field_name

        # conversion of different inputs to a common list of feature objects
        if isinstance(features, str):
            # convert the featuresclass to a list of features
            features = self._fc_to_features(dataset=features)
            if features is None:
                raise AttributeError("Feature class could not be converted to a feature set")
        elif isinstance(features, list) and len(features) > 0:
            feature = features[0]
            if isinstance(feature, Feature):
                pass
                # features passed in as a list of Feature objects

                # if "attributes" in feature.as_dict:
                #     if "geometry" in feature.as_dict:
                #         features = [Feature(feat.as_dict['geometry'], feat.as_dict['attributes']) for feat in features]
                #     else:
                #         features = [Feature(None, feat.as_dict['attributes']) for feat in features]
                # elif "geometry" in feature.as_dict:
                #     features = [Feature(feat.as_dict['geometry'], None) for feat in features]
            elif isinstance(feature, dict):
                # features passed in as a list of dicts
                if "attributes" in feature:
                    if "geometry" in feature:
                        features = [Feature(feat['geometry'], feat['attributes']) for feat in features]
                    else:
                        features = [Feature(None, feat['attributes']) for feat in features]
                elif "geometry" in feature:
                    features = [Feature(feat['geometry'], None) for feat in features]
            else:
                raise AttributeError("FeatureSet requires a list of features (as dicts or Feature objects)")

        self._features = features
        if len(features) > 0:
            feat_geom = None
            feature = features[0]

            if "geometry" in feature.as_dict: # can construct features out of tables with just attributes, no geometry
                feat_geom = feature.geometry
            elif isinstance(feature, dict):
                if "geometry" in feature:
                    feat_geom = feature['geometry']

            if feat_geom is not None:
                if spatial_reference is None:
                    if 'spatialReference' in feat_geom:
                        self._spatial_reference = feat_geom['spatialReference']

                if isinstance(feat_geom, Geometry):
                    geometry = feat_geom
                else:
                    geometry = Geometry(feat_geom)

                if geometry_type is None:
                    if isinstance(geometry, Polyline):
                        self._geometry_type = "esriGeometryPolyline"
                    elif isinstance(geometry, Polygon):
                        self._geometry_type = "esriGeometryPolygon"
                    elif isinstance(geometry, Point):
                        self._geometry_type = "esriGeometryPoint"
                    elif isinstance(geometry, MultiPoint):
                        self._geometry_type = "esriGeometryMultipoint"
                # else:
                #     raise AttributeError("Invalid geometry type") # Dont raise this error as input can be tables without geometries

            # region - build fields into a dict
            if self._fields is None or len(self._fields) == 0:
                self._fields = feature.fields  # get fields from first feature if not set

            if self._fields and isinstance(self._fields[0], str):  # as in geocoded results
                _fields = []

                for key, val in feature.attributes.items():
                    if isinstance(val, float):
                        field_type = 'esriFieldTypeDouble'
                    elif isinstance(val, int):
                        field_type = 'esriFieldTypeInteger'
                    else:
                        field_type = 'esriFieldTypeString'

                    _fields.append({'name': key,
                                    'alias': key,
                                    'type': field_type,
                                    'sqlType': 'sqlTypeOther'})

                self._fields = _fields
            # endregion

            # Try to find the object ID field if not specified
            if self._object_id_field_name is None:
                # check to see if features a dict or feature object
                if isinstance(feature, Feature):
                    # Look for OBJECTID first, if it does not exist, look for FID
                    if self._fields is None or len(self._fields) == 0:
                        self._fields = feature.fields  # get fields from first feature if not set
                    for field in feature.fields:
                        if re.search("^{0}$".format("OBJECTID"), field, re.IGNORECASE):
                            self._object_id_field_name = field
                            break
                    for field in feature.fields:
                        if re.search("^{0}$".format("FID"), field, re.IGNORECASE):
                            self._object_id_field_name = field
                            break
                else:
                    for field, _ in feature.items():
                        if re.search("^{0}$".format("OBJECTID"), field, re.IGNORECASE):
                            self._object_id_field_name = field
                            break
                    for field, _ in feature.items():
                        if re.search("^{0}$".format("FID"), field, re.IGNORECASE):
                            self._object_id_field_name = field
                            break

            # if still none, then build the objectid field
            if not self._object_id_field_name:
                obj_field = {'name': 'OBJECTID',
                             'type': 'esriFieldTypeOID',
                             'alias': 'OBJECTID',
                             'sqlType': 'sqlTypeOther'
                             }

                if len(self._fields) > 0:
                    field0 = self._fields[0]
                    if isinstance(field0, str):
                        self._fields.append('OBJECTID')
                    else:
                        self._fields.append(obj_field)
                else:
                    self._fields.append(obj_field)

                self._object_id_field_name = obj_field['name']

                counter = 0
                for feat in self._features:
                    counter += 1
                    feat.fields.append('OBJECTID')
                    if not feat.attributes: #when features have just geometry and no attributes
                        feat.attributes = {'OBJECTID':counter}
                    else:
                        feat.attributes['OBJECTID'] = counter

            # rectify Object ID field type
            if self._object_id_field_name:
                for f in self._fields:
                    if f['name'] == self._object_id_field_name:
                        if f['type'] != 'esriFieldTypeOID':
                            f['type'] = 'esriFieldTypeOID'
                        break

    # ----------------------------------------------------------------------
    def __str__(self):
        """returns object as string"""
        return json.dumps(self.value, default=_date_handler)

    def __repr__(self):
        return '<{}> {} features'.format(self.__class__.__name__, len(self.features))

    # noinspection PyUnresolvedReferences
    @staticmethod
    def _fc_to_features(dataset):
        """
        Converts a dataset to a list of feature objects, if ArcPy is available

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        dataset             Required string. Path to the featureclass.
        ===============     ====================================================================


        :return: list of feature objects
        """
        try:
            import arcpy
            arcpy_found = True
        except:
            arcpy_found = False
            raise AttributeError("ArcPy is required to create a feature set from a feature class")
        if arcpy_found:
            if not arcpy.Exists(dataset=dataset):
                raise AttributeError("Error creating FeatureSet: {0} does not exist".format(dataset))

            desc = arcpy.Describe(dataset)
            fields = [field.name for field in arcpy.ListFields(dataset) if field.type not in ['Geometry']]
            date_fields = [field.name for field in arcpy.ListFields(dataset) if field.type == 'Date']
            non_geom_fields = copy.deepcopy(fields)
            features = []
            if hasattr(desc, "shapeFieldName"):
                fields.append("SHAPE@JSON")
            del desc
            with arcpy.da.SearchCursor(dataset, fields) as rows:
                for row in rows:
                    row = list(row)
                    for date_field in date_fields:
                        if row[fields.index(date_field)] is not None:
                            row[fields.index(date_field)] = int((_date_handler(row[fields.index(date_field)])))
                    template = {
                        "attributes": dict(zip(non_geom_fields, row))
                    }
                    if "SHAPE@JSON" in fields:
                        template['geometry'] = \
                            _ujson.loads(row[fields.index("SHAPE@JSON")])

                    features.append(
                        Feature.from_dict(template)
                    )
                    del row
            return features
        return None
        # ----------------------------------------------------------------------

    @property
    def value(self):
        """returns object as dictionary"""
        val = {
            "features": [f.as_dict for f in self._features]
        }

        if self._object_id_field_name is not None:
            val["objectIdFieldName"] = self._object_id_field_name
        if self._display_field_name is not None:
            val["displayFieldName"] = self._display_field_name
        if self._global_id_field_name is not None:
            val["globalIdFieldName"] = self._global_id_field_name
        if self._spatial_reference is not None:
            val["spatialReference"] = self._spatial_reference
        if self._geometry_type is not None:
            val["geometryType"] = self._geometry_type
        if self._has_z:
            val["hasZ"] = self._has_z
        if self._has_m:
            val["hasM"] = self._has_m
        if self._fields is not None:
            val["fields"] = self._fields

        return val

        # return {
        #    "objectIdFieldName" : self._objectIdFieldName,
        #    "displayFieldName" : self._displayFieldName,
        #    "globalIdFieldName" : self._globalIdFieldName,
        #    "geometryType" : self._geometryType,
        #    "spatialReference" : self._spatialReference,
        #    "hasZ" : self._hasZ,
        #    "hasM" : self._hasM,
        #    "fields" : self._fields,
        #    "features" : [f.as_dict for f in self._features]
        # }

    # ----------------------------------------------------------------------
    @property
    def to_json(self):
        """converts the object to JSON"""
        return json.dumps(self.value, default=_date_handler)

    # ----------------------------------------------------------------------
    @property
    def to_geojson(self):
        """converts the object to GeoJSON"""
        def esri_to_geo(esrijson):
            """converts Esri Format JSON to GeoJSON"""
            def extract(feature, esri_geom_type):
                """creates a single feature"""
                item = {}
                item["type"] = "Feature"
                geom = feature["geometry"]
                geometry = {}
                geometry["type"] = get_geom_type(esri_geom_type)
                geometry["coordinates"] = get_coordinates(geom, geometry["type"])
                item["geometry"] = geometry
                item["properties"] = feature["attributes"]

                return item

            def get_geom_type(esri_type):
                """converts esri geometry types to
                GeoJSON geometry types"""
                if esri_type == "esriGeometryPoint":
                    return "Point"
                elif esri_type == "esriGeometryMultiPoint":
                    return "MultiPoint"
                elif esri_type == "esriGeometryPolyline":
                    return "MultiLineString"
                elif esri_type == "esriGeometryPolygon":
                    return "Polygon"
                else:
                    return "Point"
            def get_coordinates(geom, geom_type):
                """
                converts the Esri Geometry Structure to
                GeoJSON structure"""
                if geom_type == "Polygon":
                    return geom["rings"]
                elif geom_type == "MultiLineString":
                    return geom["paths"]
                elif geom_type == "Point":
                    return [ geom["x"], geom["y"] ]
                else:
                    return []
            geojson = {}
            features = esrijson["features"]
            esri_geom_type = esrijson["geometryType"]
            geojson["type"] = "FeatureCollection"
            feats = []
            for feat in features:
                feats.append(extract(feat, esri_geom_type))
            geojson["features"] = feats
            return geojson
        return json.dumps(esri_to_geo(self.value), default=_date_handler)

    def to_dict(self):
        """converts the object to Python dictionary"""
        return self.value

    @property
    def df(self):
        """

        **deprecated in v1.5.0 please use `sdf`**

        converts the FeatureSet to a Pandas dataframe. Requires pandas
        """
        import warnings
        warnings.warn(("The SpatialDataFrame has been deprecated. "
                       "`df` property will be removed as a future release"
                       ". Use `sdf` instead."))
        try:
            try:
                import arcpy
                arcpy_found = True
            except:
                arcpy_found = False
            from pandas.io.json import json_normalize
            from arcgis.features import SpatialDataFrame
            if len(self.features) == 0:
                import pandas as pd
                return pd.DataFrame()
            elif self.geometry_type is not None:
                if self._spatial_reference and \
                   'wkt' in self._spatial_reference.keys():
                    sr = SpatialReference(self._spatial_reference)
                elif self._spatial_reference and \
                     'wkid' in self._spatial_reference:
                    sr = SpatialReference(self._spatial_reference)
                else:
                    sr = None
                geoms = []
                attributes = []
                for feat in self.features:
                    attributes.append(feat.attributes)
                    if isinstance(feat.geometry, Geometry):
                        geoms.append(feat.geometry)
                    else:
                        g = Geometry(feat.geometry)
                        if 'spatialReference' not in g and \
                           sr is not None:
                            g['spatialReference'] = sr
                        geoms.append(g)
                    del feat
                df = json_normalize(attributes)
                df.columns = df.columns.str.replace('attributes.', '')
                return SpatialDataFrame(df, geometry=geoms, sr=sr)
            else:
                #df = pandas.DataFrame.from_dict([f.attributes for f in fs.features])
                df = json_normalize(self.value['features'])
                df.columns = df.columns.str.replace('attributes.', '')
                if self._object_id_field_name is not None:
                    df.set_index([self._object_id_field_name], inplace=True)
                else:
                    if 'OBJECTID' in df.columns:
                        df.set_index(['OBJECTID'], inplace=True)
                    elif 'FID' in df.columns:
                        df.set_index(['FID'], inplace=True)
                return df
        except ImportError:
            raise ImportError("pandas not found, please install it")

    # ----------------------------------------------------------------------
    @property
    def sdf(self):
        """
        Converts the FeatureSet to a Spatially Enabled Pandas dataframe
        """
        try:
            from arcgis.features.geo._io.serviceops import from_featureset
            return from_featureset(fset=self)
        except ImportError:
            raise Exception("Could not find the panda installation, please install Pandas and retry")
        except Exception as e:
            raise Exception("An error occurred with exporting the FeatureSet with message: %s" % str(e))
    # ----------------------------------------------------------------------
    def __iter__(self):
        """featureset iterator on features in feature set"""
        for feature in self._features:
            yield feature

    # ----------------------------------------------------------------------
    def __len__(self):
        """returns the number of features in feature set"""
        return len(self._features)

    # ----------------------------------------------------------------------
    @staticmethod
    def from_json(json_str):
        """returns a featureset from a JSON string"""
        return FeatureSet.from_dict(_ujson.loads(json_str))

    @staticmethod
    def from_dataframe(df):
        """returns a featureset from a Pandas' Data or Spatial DataFrame"""
        def _infer_type(df, col):
            """
            internal function used to get the datatypes for the feature class if
            the dataframe's _field_reference is NULL or there is a column that does
            not have a dtype assigned to it.

            Input:
             dataframe - spatialdataframe object
            Ouput:
              field type name
            """
            import numpy as np
            nn = df[col].notnull()
            nn = list(df[nn].index)
            if len(nn) > 0:
                val = df[col][nn[0]]
                if isinstance(val, str):
                    return "esriFieldTypeString"
                elif isinstance(val, tuple([int] + [np.int32])):
                    return "esriFieldTypeInteger"
                elif isinstance(val, (float, np.int64)):
                    return "esriFieldTypeDouble"
                elif isinstance(val, datetime):
                    return "esriFieldTypeDate"
            return "esriFieldTypeString"
        from ._data.geodataset import SpatialDataFrame
        import pandas as pd
        try:
            import arcpy
            HASARCPY = True
        except ImportError:
            HASARCPY = False
        features = []
        index = 0
        sr = None
        try:
            cols = [col for col in df.columns if col != df.spatial.name]
            df = df.fillna('')
        except:
            pass
        old_idx = df.index
        df.reset_index(drop=True, inplace=True)
        if isinstance(df, SpatialDataFrame):
            df_rows = df.copy()
            del df_rows['SHAPE']
            geoms = df['SHAPE'].tolist()
            sr = df.sr
        elif isinstance(df, pd.DataFrame) and \
             not df.spatial.name is None:
            fs = FeatureSet.from_dict(df.spatial.__feature_set__)
            df.set_index(old_idx,  inplace=True)
            return fs
        elif isinstance(df, pd.DataFrame):
            geoms = []
            df_rows = df.copy()
        else:
            raise ValueError("Invalid input type")
        index = 0
        for row in df_rows.to_dict('records'):
            if len(geoms) > 0:
                features.append(
                    {
                        "geometry": _ujson.loads(json.dumps(geoms[index])),
                        "attributes": row
                    })
            else:
                features.append(
                    {
                        "attributes": row
                    })
            index += 1
        fs =  FeatureSet.from_dict(featureset_dict={'features': features})
        fields = []
        for col in df_rows.columns:
            #if col not in df_rows.geometry.name:
            fields.append(
                {
                    "name" : col,
                    "type" : _infer_type(df=df_rows, col=col)
                }
            )
        fs._fields = fields
        if sr is not None:
            fs.spatial_reference = sr
        df.set_index(old_idx,  inplace=True)
        return fs
    # ----------------------------------------------------------------------
    @staticmethod
    def from_geojson(geojson):
        """
        Converts a GeoJSON Feature Collection into a FeatureSet

        """
        from warnings import warn
        def geo_to_esri(geojson):
            esri = {}

            # we already know the spatial reference we want
            # for geojson, at least in this simple case
            if 'crs' in geojson:
                warn("crs has been deprecated and will be ignored. Please"+ \
                     " see: https://tools.ietf.org/html/rfc7946#section-4 for" +\
                     " more information.")
            sr = {"wkid": 4326}

            # This will hold the geojson geometry type
            geo_type = ""
            # check for collection of features
            # and iterate as necessary
            attribute_fields = []
            if geojson["type"] == "FeatureCollection":
                features = geojson["features"]
                geo_type = features[0]["geometry"]["type"]
                attribute_fields = features[0]["properties"]

                esri_features = map(extract, features)
            else:
                attribute_fields = geojson["properties"]
                geo_type = geojson["geometry"]["type"]
                esri_features = extract(geojson)

            fields = map(extract_field, attribute_fields)
            # everything should be ready to define for the
            # esri json
            esri["geometryType"] = get_geom_type(geo_type)
            esri["spatialReference"] = sr
            esri["fields"] = list(fields)
            esri["features"] = list(esri_features)

            return esri

        def extract(feature):
            # parse out the geometry data
            geometry = get_geometry(feature)
            out_feature = {}
            out_feature["geometry"] = geometry
            out_feature["attributes"] = feature["properties"]

            return out_feature

        def extract_field(attribute):
            # now we need the fields in the properties
            a = {}
            a["alias"] = attribute
            a["name"] = attribute
            if isinstance(attribute, int):
                a["type"] = "esriFieldTypeSmallInteger"
            elif isinstance(attribute, float):
                a["type"] = "esriFieldTypeDouble"
            else:
                a["type"] = "esriFieldTypeString"
                a["length"] = 70
            return a

        def get_geom_type(geo_type):
            if geo_type == "Point":
                return "esriGeometryPoint"
            elif geo_type == "MultiPoint":
                return "esriGeometryMultiPoint"
            elif geo_type in ["LineString", "MultiLineString"]:
                return "esriGeometryPolyline"
            elif geo_type == "Polygon" or geo_type == "MultiPolygon":
                return "esriGeometryPolygon"
            else:
                return "unknown"

        def get_geometry(feature):
            # match how geometry is represented
            # based on the geojson geometry type
            geometry = {}
            geom = feature["geometry"]
            geo_type = geom["type"]
            if geo_type == "Point":
                geometry["x"] = geom["coordinates"][0]
                geometry["y"] = geom["coordinates"][1]
            elif geo_type == "Polygon":
                geometry["rings"] = [[pt for pt in reversed(g)] for g in geom['coordinates']]
            elif geo_type == "MultiPolygon":
                rings = []
                if HASARCPY:
                    if isinstance(geom, dict):
                        geometry = Geometry(geom)
                    else:
                        geom = arcpy.AsShape(geom)
                        geometry = Geometry(_ujson.loads(geom))
                else:
                    coordkey = ([d for d in geom if d.lower() == 'coordinates']
                                    or ['coordinates']).pop()
                    coordinates = geom[coordkey]
                    typekey = ([d for d in geom if d.lower() == 'type']
                                   or ['type']).pop()
                    if geom[typekey].lower() == "polygon":
                        coordinates = [coordinates]
                    part_list = []
                    for part in coordinates:
                        part_item = []
                        for idx, ring in enumerate(part):
                            if idx:
                                part_item.append(None)
                            for coord in reversed(ring):
                                part_item.append(coord)
                        if part_item:
                            part_list.append(part_item)
                    geometry["rings"] = part_list#[0]
            elif geo_type == "MultiPoint":
                geometry["points"] = [c[:] for c in geom["coordinates"]]
            elif geo_type == "LineString":
                geometry["paths"] = [[c[:] for c in geom["coordinates"]]]
            elif geo_type == "MultiLineString":
                if HASARCPY == 'rem':
                    geom = arcpy.AsShape(geom)
                    geom['spatialReference'] = {'wkid': 4326}
                    geometry = Geometry(_ujson.loads(geom))
                else:
                    coordkey = ([d for d in geom if d.lower() == 'coordinates']
                                or ['coordinates']).pop()
                    coordinates = geom[coordkey]
                    typekey = ([d for d in geom if d.lower() == 'type']
                               or ['type']).pop()
                    if geom[typekey].lower() == "linestring":
                        coordinates = [coordinates]
                    geometry["paths"] = coordinates
            if not 'spatialReference' in geometry:
                geometry['spatialReference'] = {'wkid': 4326}

            return geometry
        return FeatureSet.from_dict(geo_to_esri(geojson))
    # ----------------------------------------------------------------------
    @staticmethod
    def from_dict(featureset_dict):
        """returns a featureset from a dict"""
        features = []
        if 'fields' in featureset_dict:
            fields = featureset_dict['fields']
        else:
            fields = []
        if 'features' in featureset_dict:
            sr = featureset_dict.get('spatialReference', None)
            for feat in featureset_dict['features']:

                features.append(Feature.from_dict(feat, sr=sr))
        return FeatureSet(
            features=features, fields=fields,
            has_z=featureset_dict['hasZ'] if 'hasZ' in featureset_dict else False,
            has_m=featureset_dict['hasM'] if 'hasM' in featureset_dict else False,
            geometry_type=featureset_dict['geometryType'] if 'geometryType' in featureset_dict else None,
            object_id_field_name=featureset_dict['objectIdFieldName'] if 'objectIdFieldName' in featureset_dict else None,
            global_id_field_name=featureset_dict['globalIdFieldName'] if 'globalIdFieldName' in featureset_dict else None,
            display_field_name=featureset_dict['displayFieldName'] if 'displayFieldName' in featureset_dict else None,
            spatial_reference=featureset_dict['spatialReference'] if 'spatialReference' in featureset_dict else None)

    # ----------------------------------------------------------------------
    @property
    def spatial_reference(self):
        """gets the featureset's spatial reference"""
        return self._spatial_reference

    # ----------------------------------------------------------------------
    @spatial_reference.setter
    def spatial_reference(self, value):
        """sets the featureset's spatial reference"""
        if isinstance(value, SpatialReference):
            self._spatial_reference = value
        elif isinstance(value, int):
            self._spatial_reference = SpatialReference(wkid=value)
        elif isinstance(value, str) and \
                str(value).isdigit():
            self._spatial_reference = SpatialReference(wkid=int(value))
        else:
            self._spatial_reference = SpatialReference(value)


    # ----------------------------------------------------------------------
    @property
    def has_z(self):
        """gets/sets the Z-property"""
        return self._has_z

    # ----------------------------------------------------------------------
    @has_z.setter
    def has_z(self, value):
        """gets/sets the Z-property"""
        if isinstance(value, bool):
            self._has_z = value

    # ----------------------------------------------------------------------
    @property
    def has_m(self):
        """gets/set the M-property"""
        return self._has_m

    # ----------------------------------------------------------------------
    @has_m.setter
    def has_m(self, value):
        """gets/set the M-property"""
        if isinstance(value, bool):
            self._has_m = value

    # ----------------------------------------------------------------------
    @property
    def geometry_type(self):
        """gets/sets the geometry Type"""
        return self._geometry_type

    # ----------------------------------------------------------------------
    @geometry_type.setter
    def geometry_type(self, value):
        """gets/sets the geometry Type"""
        if value in self._allowed_geom_types:
            self._geometry_type = value

    # ----------------------------------------------------------------------
    @property
    def object_id_field_name(self):
        """gets/sets the object id field"""
        return self._object_id_field_name

    # ----------------------------------------------------------------------
    @object_id_field_name.setter
    def object_id_field_name(self, value):
        """gets/sets the object id field"""
        self._object_id_field_name = value

    # ----------------------------------------------------------------------
    @property
    def global_id_field_name(self):
        """gets/sets the globalIdFieldName"""
        return self._global_id_field_name

    # ----------------------------------------------------------------------
    @global_id_field_name.setter
    def global_id_field_name(self, value):
        """gets/sets the globalIdFieldName"""
        self._global_id_field_name = value

    # ----------------------------------------------------------------------
    @property
    def display_field_name(self):
        """gets/sets the displayFieldName"""
        return self._display_field_name

    # ----------------------------------------------------------------------
    @display_field_name.setter
    def display_field_name(self, value):
        """gets/sets the displayFieldName"""
        self._display_field_name = value

    # ----------------------------------------------------------------------
    def save(self, save_location, out_name, encoding=None):
        """
        Saves a featureset object to a feature class




        =================    ====================================================================
        **Argument**         **Description**
        -----------------    --------------------------------------------------------------------
        save_location        Required string. Path to export the FeatureSet to.
        -----------------    --------------------------------------------------------------------
        out_name             Required string. Name of the saved table.
        -----------------    --------------------------------------------------------------------
        encoding             Optional string. character encoding is used to represent a
                             repertoire of characters by some kind of encoding system. The
                             default is None.
        =================    ====================================================================


        :return: string

        """
        _, file_extension = os.path.splitext(out_name)
        if file_extension.lower() not in ['.csv', '.json'] and \
           HASARCPY == False:
            raise ImportError("ArcPy is required to export a feature class.")
        import sys
        if sys.version_info[0] == 2:
            access = 'wb+'
            kwargs = {}
            if encoding is not None:
                kwargs['encoding'] = encoding
        else:
            access = 'wt+'
            kwargs = {'newline': ''}
            if encoding is not None:
                kwargs['encoding'] = encoding

        if file_extension == ".csv":
            res = os.path.join(save_location, out_name)
            with open(res, access, **kwargs) as csv_file:
                import csv
                csv_writer = csv.writer(csv_file)
                fields = []
                # write the headers to the csv
                for field in self.fields:
                    fields.append(field['name'])
                csv_writer.writerow(fields)

                new_row = []
                # Loop through the results and save each to a row
                for feature in self:
                    new_row = []
                    for field in self.fields:
                        new_row.append(feature.get_value(field['name']))
                    csv_writer.writerow(new_row)
                csv_file.close()
            del csv_file
        elif file_extension == ".json":
            res = os.path.join(save_location, out_name)
            with open(res, access, **kwargs) as writer:
                json.dump(self.value, writer, sort_keys=True, indent=4, ensure_ascii=False)
            del writer
        else:
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, "%s.json" % uuid.uuid4().hex)
            with open(temp_file, access, **kwargs) as writer:
                json.dump(self.value,
                          writer,
                          default=_date_handler)
            del writer
            res = json_to_featureclass(json_file=temp_file,
                                       out_fc=os.path.join(save_location, out_name))
            os.remove(temp_file)
        return res

    # ----------------------------------------------------------------------
    @property
    def features(self):
        """gets the features in the FeatureSet"""
        return self._features

    # ----------------------------------------------------------------------
    @property
    def fields(self):
        """gets the fields in the FeatureSet"""
        # todo - build object id field if not found - webmaps need this
        if not self._object_id_field_name:
            obj_field = {'name': 'OBJECTID',
                         'type': 'esriFieldTypeOID',
                         'alias': 'OBJECTID',
                         'sqlType': 'sqlTypeOther'
                         }

            if len(self._fields) > 0:
                field0 = self._fields[0]
                if isinstance(field0, str):
                    self._fields.append('OBJECTID')
                else:
                    self._fields.append(obj_field)
            else:
                self._fields.append(obj_field)

            self._object_id_field_name = obj_field['name']

            counter = 0
            for feat in self.features:
                counter += 1
                feat.fields.append('OBJECTID')
                feat.attributes['OBJECTID'] = counter

        return self._fields

    # ----------------------------------------------------------------------
    @fields.setter
    def fields(self, fields):
        """sets the fields in the FeatureSet"""
        self._fields = fields


class FeatureCollection(Layer):
    """
    FeatureCollection is an object with a layer definition and a feature set.

    It is an in-memory collection of features with rendering information.

    Feature Collections can be stored as Items in the GIS, added as layers to a map or scene,
    passed as inputs to feature analysis tools, and returned as results from feature analysis tools
    if an output name for a feature layer is not specified when calling the tool.
    """

    # noinspection PyMissingConstructor
    def __init__(self, dictdata):
        self._hydrated = True
        self.properties = PropertyMap(dictdata)
        self.layer = self.properties

    @property
    def _lyr_json(self):
        return dict(self.properties)

    @property
    def _lyr_dict(self):
        return dict(self.properties)

    def __str__(self):
        return '<%s>' % type(self).__name__

    def __repr__(self):
        return '<%s>' % type(self).__name__

    def query(self):
        """
        Returns the data in this feature collection as a FeatureSet.
        Filtering by where clause is not supported for feature collections
        """
        if 'layers' in self.properties:
            if 'fields' in self.properties['layers'][0]['layerDefinition']:
                self.properties['layers'][0]['featureSet']['fields'] = \
                    self.properties['layers'][0]['layerDefinition']['fields']

            return FeatureSet.from_dict(self.properties['layers'][0]['featureSet'], )
        else:
            if 'fields' in self.properties['layerDefinition']:
                self.properties['featureSet']['fields'] = self.properties['layerDefinition']['fields']

            return FeatureSet.from_dict(self.properties['featureSet'])

    @staticmethod
    def from_featureset(fset, symbol=None, name=None):
        """
        Create a FeatureCollection object from a FeatureSet object.

        ==================     ====================================================================
        **Argument**           **Description**
        ------------------     --------------------------------------------------------------------
        fset                   Required arcgis.features.FeatureSet object.
        ------------------     --------------------------------------------------------------------
        symbol                 Optional dict. Specify your symbol as a dictionary. Symbols for points
                               can be picked from http://esri.github.io/arcgis-python-api/tools/symbol.html

                               If not specified, a default symbol will be created.
        ------------------     --------------------------------------------------------------------
        name                   Optional String. The name of the feature collection. This is used
                               when feature collections are being persisted on a WebMap. If None is
                               provided, then a random name is generated. (New at 1.6.1)
        ==================     ====================================================================
        :return:
            A FeatureCollection object.
        """
        if not isinstance(fset, FeatureSet):
            raise ValueError

        fset_dict = fset.to_dict()

        # region compose layer definition

        fc_layer_definition = {'geometryType' : fset_dict['geometryType'],
                               'fields' : fset_dict['fields'],
                               'spatialReference' : fset_dict['spatialReference'],
                               'objectIdField' : fset.object_id_field_name,
                               'type' : 'Feature Layer'}

        if not 'name' in fc_layer_definition:
            if name:
                fc_layer_definition['name'] = name.replace(" ", "_")
            else:
                fc_layer_definition['name'] = "a" + uuid.uuid4().hex[:5]

        if not 'id' in fc_layer_definition:
            fc_layer_definition['id'] = 0

        if not symbol:
            if fc_layer_definition['geometryType'] == 'esriGeometryPolyline':
                symbol = {"color": [0, 0, 0, 255],
                               "width": 1.33,
                               "type": "esriSLS",
                               "style": "esriSLSSolid"}

            elif fc_layer_definition['geometryType'] in ['esriGeometryPolygon', 'esriGeometryEnvelope']:
                symbol = {"color": [0, 0, 0, 64],
                               "outline": {
                                   "color": [0, 0, 0, 255],
                                   "width": 1.33,
                                   "type": "esriSLS",
                                   "style": "esriSLSSolid"},
                               "type": "esriSFS",
                               "style": "esriSFSSolid"}

            elif fc_layer_definition['geometryType'] in ['esriGeometryPoint', 'esriGeometryMultipoint']:
                symbol = {"angle": 0,
                               "xoffset": 0,
                               "yoffset": 12,
                               "type": "esriPMS",
                               "url": "https://esri.github.io/arcgis-python-api/notebooks/nbimages/pink.png",
                               "contentType": "image/png",
                               "width": 24,
                               "height": 24}

        fc_layer_definition['drawingInfo'] = {'renderer':{'type':'simple',
                                                          'symbol':symbol}
                                              }
        # endregion
        # compose the feature collection dict

        layers_dict = {'featureSet':{'geometryType':fset_dict['geometryType'],
                                   'features':fset_dict['features']},
                    'layerDefinition':fc_layer_definition}

        fc_dict = {'layers':[layers_dict]}

        #create a FC and return
        return FeatureCollection(fc_dict)
