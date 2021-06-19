"""
Spatial DataFrame Object developed off of the Panda's Dataframe object
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings

import arcgis
from six import string_types, integer_types


HAS_PANDAS = True
try:
    import pandas as pd
    from pandas import DataFrame, Series, Index
    import numpy
    from .base import BaseSpatialPandas
    from .geoseries import GeoSeries
except:
    HAS_PANDAS = False

    class DataFrame:
        pass

    class BaseSpatialPandas:
        pass

from arcgis.gis import GIS
from six import PY3
from six import string_types
from arcgis.geometry import _types
GEO_COLUMN_DEFAULT = "SHAPE"
GEOM_TYPES = (_types.Point, _types.MultiPoint,
              _types.Polygon,_types.Geometry,
              _types.Polyline,
              _types.BaseGeometry)

try:
    import arcpy
    from arcpy import Geometry
    HASARCPY = True
    HAS_ARCPY = True
    GEOM_TYPES = [arcpy.Point, arcpy.Polygon,
                  arcpy.Geometry, arcpy.PointGeometry,
                  arcpy.Polyline, arcpy.Multipatch,
                  arcpy.Multipoint] + list(GEOM_TYPES)
    GEOM_TYPES = tuple(GEOM_TYPES)
except:
    # warning.warn("Missing Pro will cause functionality to be limited")
    HASARCPY = False
    HAS_ARCPY = False
try:
    import shapely
    from shapely.geometry.base import BaseGeometry as _BaseGeometry
    GEOM_TYPES = [_BaseGeometry] + list(GEOM_TYPES)
    GEOM_TYPES = tuple(GEOM_TYPES)
    HASSHAPELY = True
except:
    HASSHAPELY = False

class SpatialDataFrame(BaseSpatialPandas, DataFrame):
    """
        **This class is deprecated infavor of the GeoAccessor/GeoSeriesAccessor Pattern**
    
        A Spatial Dataframe is an object to manipulate, manage and translate
        data into new forms of information for users.

        Functionality of the Spatial DataFrame is determined by the Geometry Engine
        available to the object at creation.  It will first leverage the arcpy
        geometry engine, then shapely, then it will create the geometry objects
        without any engine.

        **Scenerios**

        =================   ======================================================
        **Engine Type**     **Functionality**
        -----------------   ------------------------------------------------------
        ArcPy               Users will have the full functionality provided by the
                            API.
        -----------------   ------------------------------------------------------
        Shapely             Users get a sub-set of operations, and all properties.

                            :Valid Properties:

                            - JSON
                            - WKT
                            - WKB
                            - area
                            - centroid
                            - extent
                            - first_point
                            - hull_rectangle
                            - is_multipart
                            - label_point
                            - last_point
                            - length
                            - length3D
                            - part_count
                            - point_count
                            - true_centroid

                            :Valid Functions:

                            - boundary
                            - buffer
                            - contains
                            - convex_hull
                            - crosses
                            - difference
                            - disjoint
                            - distance_to
                            - equals
                            - generalize
                            - intersect
                            - overlaps
                            - symmetric_difference
                            - touches
                            - union
                            - within

                            Everything else will return None

        -----------------   ------------------------------------------------------
        No Engine           Values will return None by default
        =================   ======================================================



        Required Parameters:
          None
        Optional:
          :param data: panda's dataframe containing attribute information
          :param geometry: list/array/geoseries of arcgis.geometry objects
          :param sr: spatial reference of the dataframe.  This can be the factory
           code, WKT string, arcpy.SpatialReference object, or
           arcgis.SpatailReference object.
          :param gis: passing a gis.GIS object set to Pro will ensure arcpy is
           installed and a full swatch of functionality is available to
           the end user.
    """
    _internal_names = ['_data', '_cacher', '_item_cache', '_cache',
                       'is_copy', '_subtyp', '_index',
                       '_default_kind', '_default_fill_value', '_metadata',
                       '__array_struct__', '__array_interface__']
    _metadata = ['sr', '_geometry_column_name', '_gis']
    _geometry_column_name = GEO_COLUMN_DEFAULT
    #----------------------------------------------------------------------
    def __init__(self, *args, **kwargs):
        """

        .. deprecated:: 1.5.0
           see :class:`GeoAccessor` and :class:`GeoSeriesAccessor`

        .. note::
           This class is no longer maintained past version 1.4.2

        A Spatial Dataframe is an object to manipulate, manage and translate
        data into new forms of information for users.

        Required Parameters:
          None

        =====================  ===============================================================
        **optional argument**      **Description**
        ---------------------  ---------------------------------------------------------------
        data                   optional Panda's dataframe, object containing the attribute
                               information.
        ---------------------  ---------------------------------------------------------------
        index                  optional Index or array-like
                               Index to use for resulting frame. Will default to np.arange(n)
                               if no indexing information part of input data and no index
                               provided
        ---------------------  ---------------------------------------------------------------
        columns                optional Index or array-like, Column labels to use for
                               resulting frame. Will default to np.arange(n) if no column
                               labels are provided
        ---------------------  ---------------------------------------------------------------
        dtype                  dytpe, default None, Data type to force, otherwise infer
        ---------------------  ---------------------------------------------------------------
        copy                   optional boolean, default False. Copy data from inputs.
        ---------------------  ---------------------------------------------------------------
        geometry               optional list, default None, list/array/geoseries of
                               arcgis.geometry objects
        ---------------------  ---------------------------------------------------------------
        sr                     optional spatial reference of the dataframe.
        ---------------------  ---------------------------------------------------------------
        gis                    optional gis.GIS object, default None. The GIS object allowes
                               users to use non-public GIS information.
        =====================  ===============================================================

        Example: Creating SpatialDataFrame from a CSV

        df = pd.read_csv(r'D:\ipython_working_folder\joel\store_locations.csv', index_col='OBJECTID')
        geoms = []
        for i in range(0, len(df)):
            x = df.iloc[i]['X']
            y = df.iloc[i]['Y']
            geoms.append(Point({"x" : x, "y" : y, "spatialReference" : {"wkid" : 4326}}))
        sdf = arcgis.features.SpatialDataFrame(data=df, geometry=geoms)

        Example: Creating SpatialDataFrame Using List Comprehension

        coords = [[1,2], [3,4]]
        sdf = SpatialDataFrame(df,
                     geometry=[arcgis.geometry.Geometry({'x':r[0],
                     'y':r[1], 'spatialReference':{'wkid':4326}}) for r in coords])

        .. Note: When passing in a geometry to the SpatialDataFrame, always assign it to the parameter geometry=<var>

        Example: Creating From Feature Class

        sdf = SpatialDataFrame.from_featureclass(r"c:\temp\data.gdb\cities)

        Example: Create A SpatialDataFrame from a Service

        gis = GIS(username="user1", password="password2")
        item = gis.content.search("Roads")[0]
        feature_layer = item.layers[0]
        sdf = SpatialDataFrame.from_layer(feature_layer)

        """
        warnings.warn("SpatialDataFrame has been deprecated.  Please switch to the GeoAccessor/GeoSeriesAccessor.")

        if not HAS_PANDAS:
            warnings.warn("pandas and numpy are required for SpatialDataFrame.")
            warnings.warn("Please install them.")
        gis = kwargs.pop('gis', arcgis.env.active_gis)
        self._gis = gis
        sr = self._sr(kwargs.pop('sr', 4326))
        geometry = kwargs.pop('geometry', None)
        super(SpatialDataFrame, self).__init__(*args, **kwargs)

        if isinstance(sr, _types.SpatialReference):
            self.sr = sr
        elif isinstance(sr, integer_types):
            self.sr = _types.SpatialReference({'wkid' : sr})
        elif isinstance(sr, string_types):
            self.sr = _types.SpatialReference({'wkt' : sr})
        elif hasattr(sr, 'factoryCode'):
            self.sr = _types.SpatialReference({'wkid' : sr.factoryCode})
        elif hasattr(sr, 'exportToString'):
            self.sr = _types.SpatialReference({'wkt' : sr.exportToString()})
        elif not sr is None:
            raise ValueError("sr (spatial reference) must be a _types.SpatialReference object")
        else:
            self.sr = None
        if geometry is not None:
            self.set_geometry(geometry, inplace=True)
        elif 'SHAPE' in self.columns:
            if isinstance(self['SHAPE'], (GeoSeries, pd.Series)) and self['SHAPE'].dtype.name == 'object':
                if all(isinstance(x, _types.Geometry) for x in self[self._geometry_column_name]) == False:
                    geometry = [_types.Geometry(g) for g in self['SHAPE'].tolist()]
                    del self['SHAPE']
                    self.set_geometry(geometry, inplace=True)
        if self.sr is None:
            try:
                self.sr = self.geometry[self.geometry.first_valid_index()].spatialReference
            except:
                self.sr = self._sr(sr)
        self._delete_index()
    #----------------------------------------------------------------------
    @property
    def _constructor(self):
        """constructor for class as per Pandas' github page"""
        return SpatialDataFrame
    #----------------------------------------------------------------------
    def info(self, verbose=None, buf=None, max_cols=None, memory_usage=None, null_counts=None):
        """
        Concise summary of a DataFrame.

        Parameters
        ----------
        verbose : {None, True, False}, optional
            Whether to print the full summary.
            None follows the `display.max_info_columns` setting.
            True or False overrides the `display.max_info_columns` setting.
        buf : writable buffer, defaults to sys.stdout
        max_cols : int, default None
            Determines whether full summary or short summary is printed.
            None follows the `display.max_info_columns` setting.
        memory_usage : boolean/string, default None
            Specifies whether total memory usage of the DataFrame
            elements (including index) should be displayed. None follows
            the `display.memory_usage` setting. True or False overrides
            the `display.memory_usage` setting. A value of 'deep' is equivalent
            of True, with deep introspection. Memory usage is shown in
            human-readable units (base-2 representation).
        null_counts : boolean, default None
            Whether to show the non-null counts

            - If None, then only show if the frame is smaller than
              max_info_rows and max_info_columns.
            - If True, always show counts.
            - If False, never show counts.

        """
        cols = list(self.columns)
        cols.pop(cols.index("SHAPE"))
        return self[cols].info(verbose, buf, max_cols,
                               memory_usage, null_counts)
    #----------------------------------------------------------------------
    def _sr(self, sr):
        """sets the spatial reference"""
        if isinstance(sr, _types.SpatialReference):
            return sr
        elif isinstance(sr, dict):
            return _types.SpatialReference(sr)
        elif isinstance(sr, integer_types):
            return _types.SpatialReference({'wkid' : sr})
        elif isinstance(sr, string_types):
            return _types.SpatialReference({'wkt' : sr})
        elif hasattr(sr, 'factoryCode'):
            return _types.SpatialReference({'wkid' : sr.factoryCode})
        elif hasattr(sr, 'exportToString'):
            return _types.SpatialReference({'wkt' : sr.exportToString()})
        elif not sr is None:
            raise ValueError("sr (spatial reference) must be a _types.SpatialReference object")
        else:
            return None
    #----------------------------------------------------------------------
    @property
    def __feature_set__(self):
        """returns a dictionary representation of an Esri FeatureSet"""
        import numpy as np
        import datetime
        import time
        cols_norm = [col for col in self.columns]
        cols_lower = [col.lower() for col in self.columns]
        fields = []
        features = []
        date_fields = []
        _geom_types = {
            arcgis.geometry._types.Point :  "esriGeometryPoint",
            arcgis.geometry._types.Polyline : "esriGeometryPolyline",
            arcgis.geometry._types.MultiPoint : "esriGeometryMultipoint",
            arcgis.geometry._types.Polygon : "esriGeometryPolygon"
        }
        if self.sr is None:
            sr = {'wkid' : 4326}
        else:
            sr = self.sr
        fs = {
            "objectIdFieldName" : "",
            "globalIdFieldName" : "",
            "displayFieldName" : "",
            "geometryType" : _geom_types[type(self.geometry[self.geometry.first_valid_index()])],
            "spatialReference" : sr,
            "fields" : [],
            "features" : []
        }
        if 'objectid' in cols_lower:
            fs['objectIdFieldName'] = cols_norm[cols_lower.index('objectid')]
            fs['displayFieldName'] = cols_norm[cols_lower.index('objectid')]
        elif 'fid' in cols_lower:
            fs['objectIdFieldName'] = cols_norm[cols_lower.index('fid')]
            fs['displayFieldName'] = cols_norm[cols_lower.index('fid')]
        elif 'oid' in cols_lower:
            fs['objectIdFieldName'] = cols_norm[cols_lower.index('oid')]
            fs['displayFieldName'] = cols_norm[cols_lower.index('oid')]
        else:
            self['OBJECTID'] = list(range(1, self.shape[0] + 1))
            res = self.__feature_set__
            del self['OBJECTID']
            return res
        if 'objectIdFieldName' in fs:
            fields.append({
                "name" : fs['objectIdFieldName'],
                "type" : "esriFieldTypeOID",
                "alias" : fs['objectIdFieldName']
            })
            cols_norm.pop(cols_norm.index(fs['objectIdFieldName']))
        if 'globalIdFieldName' in fs and len(fs['globalIdFieldName']) > 0:
            fields.append({
                "name" : fs['globalIdFieldName'],
                "type" : "esriFieldTypeGlobalID",
                "alias" : fs['globalIdFieldName']
            })
            cols_norm.pop(cols_norm.index(fs['globalIdFieldName']))
        elif 'globalIdFieldName' in fs and \
             len(fs['globalIdFieldName']) == 0:
            del fs['globalIdFieldName']
        if self._geometry_column_name in cols_norm:
            cols_norm.pop(cols_norm.index(self._geometry_column_name))
        for col in cols_norm:
            try:
                idx = self[col].first_valid_index()
                col_val = self[col].loc[idx]
            except:
                col_val = ""
            if isinstance(col_val, (str, np.str)):
                l = self[col].str.len().max()
                if str(l) == 'nan':
                    l = 255

                fields.append({
                    "name" : col,
                    "type" : "esriFieldTypeString",
                    "length" : int(l),
                    "alias" : col
                })
                if fs['displayFieldName'] == "":
                    fs['displayFieldName'] = col
            elif isinstance(col_val, (datetime.datetime,
                                      pd.Timestamp,
                                      np.datetime64,
                                      pd.datetime)):
                fields.append({
                    "name" : col,
                    "type" : "esriFieldTypeDate",
                    "alias" : col
                })
                date_fields.append(col)
            elif isinstance(col_val, (np.int32, np.int16, np.int8)):
                fields.append({
                    "name" : col,
                    "type" : "esriFieldTypeSmallInteger",
                    "alias" : col
                })
            elif isinstance(col_val, (int, np.int, np.int64)):
                fields.append({
                    "name" : col,
                    "type" : "esriFieldTypeInteger",
                    "alias" : col
                })
            elif isinstance(col_val, (float, np.float64)):
                fields.append({
                    "name" : col,
                    "type" : "esriFieldTypeDouble",
                    "alias" : col
                })
            elif isinstance(col_val, (np.float32)):
                fields.append({
                    "name" : col,
                    "type" : "esriFieldTypeSingle",
                    "alias" : col
                })
        fs['fields'] = fields
        for row in self.to_dict('records'):
            geom = {}
            if self._geometry_column_name in row:
                geom = row[self._geometry_column_name]
                del row[self._geometry_column_name]
            for f in date_fields:
                try:
                    row[f] = int(row[f].to_pydatetime().timestamp() * 1000)
                except:
                    row[f] = None
            features.append(
                {
                    "geometry" : dict(geom),
                    "attributes" : row
                }
            )
            del row
            del geom
        fs['features'] = features
        return fs
    #----------------------------------------------------------------------
    @property
    def __geo_interface__(self):
        """returns the object as an Feature Collection JSON string"""
        if HASARCPY:
            template = {
                "type": "FeatureCollection",
                "features": []
            }
            geom_type = self.geometry_type
            if geom_type.lower() == "point":
                geom_type = "Point"
            elif geom_type.lower() == "polyline":
                geom_type = "LineString"
            elif geom_type.lower() == "polygon":
                geom_type = "Polygon"
            df_copy = self.copy(deep=True)
            df_copy['geom_json'] = self.geometry.JSON
            df_copy['SHAPE'] = df_copy['geom_json']
            del df_copy['geom_json']
            for index, row in df_copy.iterrows():
                geom = row['SHAPE']
                del row['SHAPE']
                template['features'].append(
                    {"type" : geom_type,
                     "geometry" : pd.io.json.loads(geom),
                     "attributes":row}
                )
            return pd.io.json.dumps(template)
    @property
    def geoextent(self):
        """returns the extent of the spatial dataframe"""
        return self.series_extent
    #----------------------------------------------------------------------
    def __getstate__(self):
        meta = {k: getattr(self, k, None) for k in self._metadata}
        return dict(_data=self._data, _typ=self._typ,
                    _metadata=self._metadata, **meta)
    #----------------------------------------------------------------------
    def __setattr__(self, attr, val):
        if attr.lower() in ['geometry', 'shape', 'shape@']:
            object.__setattr__(self, attr, val)
        else:
            super(SpatialDataFrame, self).__setattr__(attr, val)
    #----------------------------------------------------------------------
    def _get_geometry(self):
        """returns the geometry series"""
        if self._geometry_column_name not in self.columns:
            raise AttributeError("Geometry Column Not Present: %s" % self._geometry_column_name)
        return self[self._geometry_column_name]
    #----------------------------------------------------------------------
    def _set_geometry(self, col):
        """sets the geometry for the panda's dataframe"""
        if isinstance(col, (GeoSeries, list, numpy.array, numpy.ndarray, Series)):
            self.set_geometry(col, inplace=True)
        else:
            raise ValueError("Must be a list, np.array, or GeoSeries")
    #----------------------------------------------------------------------
    geometry = property(fget=_get_geometry,
                        fset=_set_geometry,
                        fdel=None,
                        doc="Get/Set the geometry data for SpatialDataFrame")
    #----------------------------------------------------------------------
    def __finalize__(self, other, method=None, **kwargs):
        """propagate metadata from other to self """
        # merge operation: using metadata of the left object
        if method == 'merge':
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other.left, name, None))
        # concat operation: using metadata of the first object
        elif method == 'concat':
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other.objs[0], name, None))
        else:
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other, name, None))
        return self
    #----------------------------------------------------------------------
    def copy(self, deep=True):
        """
        Make a copy of this SpatialDataFrame object
        Parameters:

        :deep: boolean, default True
               Make a deep copy, i.e. also copy data
        Returns:
         :copy: of SpatialDataFrame
        """
        data = self._data
        if deep:
            data = data.copy()
        return SpatialDataFrame(data, sr=self.sr).__finalize__(self)
    #----------------------------------------------------------------------
    def plot(self, *args, **kwargs):
        """
        Plot draws the data on a web map. The user can describe in simple terms how to
        renderer spatial data using symbol.  To make the process simpler a palette
        for which colors are drawn from can be used instead of explicit colors.


        ======================  =========================================================
        **Explicit Argument**   **Description**
        ----------------------  ---------------------------------------------------------
        df                      required SpatialDataFrame or GeoSeries. This is the data
                                to map.
        ----------------------  ---------------------------------------------------------
        map_widget              optional WebMap object. This is the map to display the
                                data on.
        ----------------------  ---------------------------------------------------------
        palette                 optional string/dict.  Color mapping.  For simple renderer,
                                just provide a string.  For more robust renderers like
                                unique renderer, a dictionary can be given.
        ----------------------  ---------------------------------------------------------
        renderer_type           optional string.  Determines the type of renderer to use
                                for the provided dataset. The default is 's' which is for
                                simple renderers.

                                Allowed values:

                                + 's' - is a simple renderer that uses one symbol only.
                                + 'u' - unique renderer symbolizes features based on one
                                        or more matching string attributes.
                                + 'c' - A class breaks renderer symbolizes based on the
                                        value of some numeric attribute.
                                + 'h' - heatmap renders point data into a raster
                                        visualization that emphasizes areas of higher
                                        density or weighted values.
        ----------------------  ---------------------------------------------------------
        symbol_style            optional string. This is the type of symbol the user
                                needs to create.  Valid inputs are: simple, picture, text,
                                or carto.  The default is simple.
        ----------------------  ---------------------------------------------------------
        symbol_type             optional string. This is the symbology used by the
                                geometry.  For example 's' for a Line geometry is a solid
                                line. And '-' is a dash line.

                                Allowed symbol types based on geometries:

                                **Point Symbols**

                                 + 'o' - Circle (default)
                                 + '+' - Cross
                                 + 'D' - Diamond
                                 + 's' - Square
                                 + 'x' - X

                                 **Polyline Symbols**

                                 + 's' - Solid (default)
                                 + '-' - Dash
                                 + '-.' - Dash Dot
                                 + '-..' - Dash Dot Dot
                                 + '.' - Dot
                                 + '--' - Long Dash
                                 + '--.' - Long Dash Dot
                                 + 'n' - Null
                                 + 's-' - Short Dash
                                 + 's-.' - Short Dash Dot
                                 + 's-..' - Short Dash Dot Dot
                                 + 's.' - Short Dot

                                 **Polygon Symbols**

                                 + 's' - Solid Fill (default)
                                 + '\' - Backward Diagonal
                                 + '/' - Forward Diagonal
                                 + '|' - Vertical Bar
                                 + '-' - Horizontal Bar
                                 + 'x' - Diagonal Cross
                                 + '+' - Cross

        ----------------------  ---------------------------------------------------------
        col                     optional string/list. Field or fields used for heatmap,
                                class breaks, or unique renderers.
        ----------------------  ---------------------------------------------------------
        palette                 optional string. The color map to draw from in order to
                                visualize the data.  The default palette is 'jet'. To
                                get a visual representation of the allowed color maps,
                                use the **display_colormaps** method.
        ----------------------  ---------------------------------------------------------
        alpha                   optional float.  This is a value between 0 and 1 with 1
                                being the default value.  The alpha sets the transparancy
                                of the renderer when applicable.
        ======================  =========================================================

        **Render Syntax**

        The render syntax allows for users to fully customize symbolizing the data.

        **Simple Renderer**

        A simple renderer is a renderer that uses one symbol only.

        ======================  =========================================================
        **Optional Argument**   **Description**
        ----------------------  ---------------------------------------------------------
        symbol_style            optional string. This is the type of symbol the user
                                needs to create.  Valid inputs are: simple, picture, text,
                                or carto.  The default is simple.
        ----------------------  ---------------------------------------------------------
        symbol_type             optional string. This is the symbology used by the
                                geometry.  For example 's' for a Line geometry is a solid
                                line. And '-' is a dash line.

                                **Point Symbols**

                                + 'o' - Circle (default)
                                + '+' - Cross
                                + 'D' - Diamond
                                + 's' - Square
                                + 'x' - X

                                **Polyline Symbols**

                                + 's' - Solid (default)
                                + '-' - Dash
                                + '-.' - Dash Dot
                                + '-..' - Dash Dot Dot
                                + '.' - Dot
                                + '--' - Long Dash
                                + '--.' - Long Dash Dot
                                + 'n' - Null
                                + 's-' - Short Dash
                                + 's-.' - Short Dash Dot
                                + 's-..' - Short Dash Dot Dot
                                + 's.' - Short Dot

                                **Polygon Symbols**

                                + 's' - Solid Fill (default)
                                + '\' - Backward Diagonal
                                + '/' - Forward Diagonal
                                + '|' - Vertical Bar
                                + '-' - Horizontal Bar
                                + 'x' - Diagonal Cross
                                + '+' - Cross
        ----------------------  ---------------------------------------------------------
        description             Description of the renderer.
        ----------------------  ---------------------------------------------------------
        rotation_expression     A constant value or an expression that derives the angle
                                of rotation based on a feature attribute value. When an
                                attribute name is specified, it's enclosed in square
                                brackets.
        ----------------------  ---------------------------------------------------------
        rotation_type           String value which controls the origin and direction of
                                rotation on point features. If the rotationType is
                                defined as arithmetic, the symbol is rotated from East in
                                a counter-clockwise direction where East is the 0 degree
                                axis. If the rotationType is defined as geographic, the
                                symbol is rotated from North in a clockwise direction
                                where North is the 0 degree axis.

                                Must be one of the following values:

                                + arithmetic
                                + geographic

        ----------------------  ---------------------------------------------------------
        visual_variables        An array of objects used to set rendering properties.
        ======================  =========================================================

        **Heatmap Renderer**

        The HeatmapRenderer renders point data into a raster visualization that emphasizes
        areas of higher density or weighted values.

        ======================  =========================================================
        **Optional Argument**   **Description**
        ----------------------  ---------------------------------------------------------
        blur_radius             The radius (in pixels) of the circle over which the
                                majority of each point's value is spread.
        ----------------------  ---------------------------------------------------------
        field                   This is optional as this renderer can be created if no
                                field is specified. Each feature gets the same
                                value/importance/weight or with a field where each
                                feature is weighted by the field's value.
        ----------------------  ---------------------------------------------------------
        max_intensity           The pixel intensity value which is assigned the final
                                color in the color ramp.
        ----------------------  ---------------------------------------------------------
        min_intensity           The pixel intensity value which is assigned the initial
                                color in the color ramp.
        ----------------------  ---------------------------------------------------------
        ratio                   A number between 0-1. Describes what portion along the
                                gradient the colorStop is added.
        ======================  =========================================================

        **Unique Renderer**

        This renderer symbolizes features based on one or more matching string attributes.

        ======================  =========================================================
        **Optional Argument**   **Description**
        ----------------------  ---------------------------------------------------------
        background_fill_symbol  A symbol used for polygon features as a background if the
                                renderer uses point symbols, e.g. for bivariate types &
                                size rendering. Only applicable to polygon layers.
                                PictureFillSymbols can also be used outside of the Map
                                Viewer for Size and Predominance and Size renderers.
        ----------------------  ---------------------------------------------------------
        default_label           Default label for the default symbol used to draw
                                unspecified values.
        ----------------------  ---------------------------------------------------------
        default_symbol          Symbol used when a value cannot be matched.
        ----------------------  ---------------------------------------------------------
        col                     String or List of Strings. Attribute field(s) the
                                renderer uses to match values.
        ----------------------  ---------------------------------------------------------
        field_delimiter         String inserted between the values if multiple attribute
                                fields are specified.
        ----------------------  ---------------------------------------------------------
        rotation_expression     A constant value or an expression that derives the angle
                                of rotation based on a feature attribute value. When an
                                attribute name is specified, it's enclosed in square
                                brackets. Rotation is set using a visual variable of type
                                rotation info with a specified field or value expression
                                property.
        ----------------------  ---------------------------------------------------------
        rotation_type           String property which controls the origin and direction
                                of rotation. If the rotation type is defined as
                                arithmetic the symbol is rotated from East in a
                                counter-clockwise direction where East is the 0 degree
                                axis. If the rotation type is defined as geographic, the
                                symbol is rotated from North in a clockwise direction
                                where North is the 0 degree axis.
                                Must be one of the following values:

                                + arithmetic
                                + geographic

        ----------------------  ---------------------------------------------------------
        arcade_expression       An Arcade expression evaluating to either a string or a
                                number.
        ----------------------  ---------------------------------------------------------
        arcade_title            The title identifying and describing the associated
                                Arcade expression as defined in the valueExpression
                                property.
        ----------------------  ---------------------------------------------------------
        visual_variables        An array of objects used to set rendering properties.
        ======================  =========================================================

        **Class Breaks Renderer**

        A class breaks renderer symbolizes based on the value of some numeric attribute.

        ======================  =========================================================
        **Optional Argument**   **Description**
        ----------------------  ---------------------------------------------------------
        background_fill_symbol  A symbol used for polygon features as a background if the
                                renderer uses point symbols, e.g. for bivariate types &
                                size rendering. Only applicable to polygon layers.
                                PictureFillSymbols can also be used outside of the Map
                                Viewer for Size and Predominance and Size renderers.
        ----------------------  ---------------------------------------------------------
        default_label           Default label for the default symbol used to draw
                                unspecified values.
        ----------------------  ---------------------------------------------------------
        default_symbol          Symbol used when a value cannot be matched.
        ----------------------  ---------------------------------------------------------
        method                  Determines the classification method that was used to
                                generate class breaks.

                                Must be one of the following values:

                                + esriClassifyDefinedInterval
                                + esriClassifyEqualInterval
                                + esriClassifyGeometricalInterval
                                + esriClassifyNaturalBreaks
                                + esriClassifyQuantile
                                + esriClassifyStandardDeviation
                                + esriClassifyManual

        ----------------------  ---------------------------------------------------------
        field                   Attribute field used for renderer.
        ----------------------  ---------------------------------------------------------
        min_value               The minimum numeric data value needed to begin class
                                breaks.
        ----------------------  ---------------------------------------------------------
        normalization_field     Used when normalizationType is field. The string value
                                indicating the attribute field by which the data value is
                                normalized.
        ----------------------  ---------------------------------------------------------
        normalization_total     Used when normalizationType is percent-of-total, this
                                number property contains the total of all data values.
        ----------------------  ---------------------------------------------------------
        normalization_type      Determine how the data was normalized.

                                Must be one of the following values:

                                + esriNormalizeByField
                                + esriNormalizeByLog
                                + esriNormalizeByPercentOfTotal
        ----------------------  ---------------------------------------------------------
        rotation_expression     A constant value or an expression that derives the angle
                                of rotation based on a feature attribute value. When an
                                attribute name is specified, it's enclosed in square
                                brackets.
        ----------------------  ---------------------------------------------------------
        rotation_type           A string property which controls the origin and direction
                                of rotation. If the rotation_type is defined as
                                arithmetic, the symbol is rotated from East in a
                                couter-clockwise direction where East is the 0 degree
                                axis. If the rotationType is defined as geographic, the
                                symbol is rotated from North in a clockwise direction
                                where North is the 0 degree axis.

                                Must be one of the following values:

                                + arithmetic
                                + geographic

        ----------------------  ---------------------------------------------------------
        arcade_expression       An Arcade expression evaluating to a number.
        ----------------------  ---------------------------------------------------------
        arcade_title            The title identifying and describing the associated
                                Arcade expression as defined in the arcade_expression
                                property.
        ----------------------  ---------------------------------------------------------
        visual_variables        An object used to set rendering options.
        ======================  =========================================================



        **Symbol Syntax**

        =======================  =========================================================
        **Optional Argument**    **Description**
        -----------------------  ---------------------------------------------------------
        symbol_style             optional string. This is the type of symbol the user
                                 needs to create.  Valid inputs are: simple, picture, text,
                                 or carto.  The default is simple.
        -----------------------  ---------------------------------------------------------
        symbol_type              optional string. This is the symbology used by the
                                 geometry.  For example 's' for a Line geometry is a solid
                                 line. And '-' is a dash line.

                                 **Point Symbols**

                                 + 'o' - Circle (default)
                                 + '+' - Cross
                                 + 'D' - Diamond
                                 + 's' - Square
                                 + 'x' - X

                                 **Polyline Symbols**

                                 + 's' - Solid (default)
                                 + '-' - Dash
                                 + '-.' - Dash Dot
                                 + '-..' - Dash Dot Dot
                                 + '.' - Dot
                                 + '--' - Long Dash
                                 + '--.' - Long Dash Dot
                                 + 'n' - Null
                                 + 's-' - Short Dash
                                 + 's-.' - Short Dash Dot
                                 + 's-..' - Short Dash Dot Dot
                                 + 's.' - Short Dot

                                 **Polygon Symbols**

                                 + 's' - Solid Fill (default)
                                 + '\' - Backward Diagonal
                                 + '/' - Forward Diagonal
                                 + '|' - Vertical Bar
                                 + '-' - Horizontal Bar
                                 + 'x' - Diagonal Cross
                                 + '+' - Cross
        -----------------------  ---------------------------------------------------------
        cmap                     optional string or list.  This is the color scheme a user
                                 can provide if the exact color is not needed, or a user
                                 can provide a list with the color defined as:
                                 [red, green blue, alpha]. The values red, green, blue are
                                 from 0-255 and alpha is a float value from 0 - 1.
                                 The default value is 'jet' color scheme.
        -----------------------  ---------------------------------------------------------
        cstep                    optional integer.  If provided, its the color location on
                                 the color scheme.
        =======================  =========================================================

        **Simple Symbols**

        This is a list of optional parameters that can be given for point, line or
        polygon geometries.

        ====================  =========================================================
        **Argument**          **Description**
        --------------------  ---------------------------------------------------------
        marker_size           optional float.  Numeric size of the symbol given in
                              points.
        --------------------  ---------------------------------------------------------
        marker_angle          optional float. Numeric value used to rotate the symbol.
                              The symbol is rotated counter-clockwise. For example,
                              The following, angle=-30, in will create a symbol rotated
                              -30 degrees counter-clockwise; that is, 30 degrees
                              clockwise.
        --------------------  ---------------------------------------------------------
        marker_xoffset        Numeric value indicating the offset on the x-axis in points.
        --------------------  ---------------------------------------------------------
        marker_yoffset        Numeric value indicating the offset on the y-axis in points.
        --------------------  ---------------------------------------------------------
        line_width            optional float. Numeric value indicating the width of the line in points
        --------------------  ---------------------------------------------------------
        outline_style         Optional string. For polygon point, and line geometries , a
                              customized outline type can be provided.

                              Allowed Styles:

                              + 's' - Solid (default)
                              + '-' - Dash
                              + '-.' - Dash Dot
                              + '-..' - Dash Dot Dot
                              + '.' - Dot
                              + '--' - Long Dash
                              + '--.' - Long Dash Dot
                              + 'n' - Null
                              + 's-' - Short Dash
                              + 's-.' - Short Dash Dot
                              + 's-..' - Short Dash Dot Dot
                              + 's.' - Short Dot
        --------------------  ---------------------------------------------------------
        outline_color         optional string or list.  This is the same color as the
                              cmap property, but specifically applies to the outline_color.
        ====================  =========================================================

        **Picture Symbol**

        This type of symbol only applies to Points, MultiPoints and Polygons.

        ====================  =========================================================
        **Argument**          **Description**
        --------------------  ---------------------------------------------------------
        marker_angle          Numeric value that defines the number of degrees ranging
                              from 0-360, that a marker symbol is rotated. The rotation
                              is from East in a counter-clockwise direction where East
                              is the 0 axis.
        --------------------  ---------------------------------------------------------
        marker_xoffset        Numeric value indicating the offset on the x-axis in points.
        --------------------  ---------------------------------------------------------
        marker_yoffset        Numeric value indicating the offset on the y-axis in points.
        --------------------  ---------------------------------------------------------
        height                Numeric value used if needing to resize the symbol. Specify a value in points. If images are to be displayed in their original size, leave this blank.
        --------------------  ---------------------------------------------------------
        width                 Numeric value used if needing to resize the symbol. Specify a value in points. If images are to be displayed in their original size, leave this blank.
        --------------------  ---------------------------------------------------------
        url                   String value indicating the URL of the image. The URL should be relative if working with static layers. A full URL should be used for map service dynamic layers. A relative URL can be dereferenced by accessing the map layer image resource or the feature layer image resource.
        --------------------  ---------------------------------------------------------
        image_data            String value indicating the base64 encoded data.
        --------------------  ---------------------------------------------------------
        xscale                Numeric value indicating the scale factor in x direction.
        --------------------  ---------------------------------------------------------
        yscale                Numeric value indicating the scale factor in y direction.
        --------------------  ---------------------------------------------------------
        outline_color         optional string or list.  This is the same color as the
                              cmap property, but specifically applies to the outline_color.
        --------------------  ---------------------------------------------------------
        outline_style         Optional string. For polygon point, and line geometries , a
                              customized outline type can be provided.

                              Allowed Styles:

                              + 's' - Solid (default)
                              + '-' - Dash
                              + '-.' - Dash Dot
                              + '-..' - Dash Dot Dot
                              + '.' - Dot
                              + '--' - Long Dash
                              + '--.' - Long Dash Dot
                              + 'n' - Null
                              + 's-' - Short Dash
                              + 's-.' - Short Dash Dot
                              + 's-..' - Short Dash Dot Dot
                              + 's.' - Short Dot
        --------------------  ---------------------------------------------------------
        outline_color         optional string or list.  This is the same color as the
                              cmap property, but specifically applies to the outline_color.
        --------------------  ---------------------------------------------------------
        line_width            optional float. Numeric value indicating the width of the line in points
        ====================  =========================================================

        **Text Symbol**

        This type of symbol only applies to Points, MultiPoints and Polygons.

        ====================  =========================================================
        **Argument**          **Description**
        --------------------  ---------------------------------------------------------
        font_decoration       The text decoration. Must be one of the following values:
                              - line-through
                              - underline
                              - none
        --------------------  ---------------------------------------------------------
        font_family           Optional string. The font family.
        --------------------  ---------------------------------------------------------
        font_size             Optional float. The font size in points.
        --------------------  ---------------------------------------------------------
        font_style            Optional string. The text style.
                              - italic
                              - normal
                              - oblique
        --------------------  ---------------------------------------------------------
        font_weight           Optional string. The text weight.
                              Must be one of the following values:
                              - bold
                              - bolder
                              - lighter
                              - normal
        --------------------  ---------------------------------------------------------
        background_color      optional string/list. Background color is represented as
                              a four-element array or string of a color map.
        --------------------  ---------------------------------------------------------
        halo_color            Optional string/list. Color of the halo around the text.
                              The default is None.
        --------------------  ---------------------------------------------------------
        halo_size             Optional integer/float. The point size of a halo around
                              the text symbol.
        --------------------  ---------------------------------------------------------
        horizontal_alignment  optional string. One of the following string values
                              representing the horizontal alignment of the text.
                              Must be one of the following values:
                              - left
                              - right
                              - center
                              - justify
        --------------------  ---------------------------------------------------------
        kerning               optional boolean. Boolean value indicating whether to
                              adjust the spacing between characters in the text string.
        --------------------  ---------------------------------------------------------
        line_color            optional string/list. Outline color is represented as
                              a four-element array or string of a color map.
        --------------------  ---------------------------------------------------------
        line_width            optional integer/float. Outline size.
        --------------------  ---------------------------------------------------------
        marker_angle          optional int. A numeric value that defines the number of
                              degrees (0 to 360) that a text symbol is rotated. The
                              rotation is from East in a counter-clockwise direction
                              where East is the 0 axis.
        --------------------  ---------------------------------------------------------
        marker_xoffset        optional int/float.Numeric value indicating the offset
                              on the x-axis in points.
        --------------------  ---------------------------------------------------------
        marker_yoffset        optional int/float.Numeric value indicating the offset
                              on the x-axis in points.
        --------------------  ---------------------------------------------------------
        right_to_left         optional boolean. Set to true if using Hebrew or Arabic
                              fonts.
        --------------------  ---------------------------------------------------------
        rotated               optional boolean. Boolean value indicating whether every
                              character in the text string is rotated.
        --------------------  ---------------------------------------------------------
        text                  Required string.  Text Value to display next to geometry.
        --------------------  ---------------------------------------------------------
        vertical_alignment    Optional string. One of the following string values
                              representing the vertical alignment of the text.
                              Must be one of the following values:
                              - top
                              - bottom
                              - middle
                              - baseline
        ====================  =========================================================

        **Cartographic Symbol**

        This type of symbol only applies to line geometries.

        ====================  =========================================================
        **Argument**          **Description**
        --------------------  ---------------------------------------------------------
        line_width            optional float. Numeric value indicating the width of the line in points
        --------------------  ---------------------------------------------------------
        cap                   Optional string.  The cap style.
        --------------------  ---------------------------------------------------------
        join                  Optional string. The join style.
        --------------------  ---------------------------------------------------------
        miter_limit           Optional string. Size threshold for showing mitered line joins.
        ====================  =========================================================

        The kwargs parameter accepts all parameters of the create_symbol method and the
        create_renderer method.


        """
        if ('kind' in kwargs and \
           kwargs['kind'] == 'map') or \
           (len(args) > 3 and args[3] == 'map'):
            from arcgis.features._data.geodataset.viz import plot
            has_wm = True
            wm = kwargs.pop('map_widget', None)
            if wm is None:
                has_wm = False
                wm = GIS().map()
            if has_wm:
                plot(df=self,
                     map_widget=wm,
                     name=kwargs.pop('name', "Feature Collection Layer"),
                     renderer_type=kwargs.pop("renderer_type", None),
                     symbol_type=kwargs.pop('symbol_type', None),
                     symbol_style=kwargs.pop('symbol_style', None),
                     col=kwargs.pop('col', None),
                     colors=kwargs.pop('cmap', None) or kwargs.pop('colors', None) or kwargs.pop('pallette', 'jet'),
                     alpha=kwargs.pop('alpha', 1),
                     **kwargs)
                return True
            else:
                return plot(df=self,
                            map_widget=wm,
                            name=kwargs.pop('name', "Feature Collection Layer"),
                            renderer_type=kwargs.pop("renderer_type", None),
                            symbol_type=kwargs.pop('symbol_type', None),
                            symbol_style=kwargs.pop('symbol_style', None),
                            col=kwargs.pop('col', None),
                            colors=kwargs.pop('cmap', None) or kwargs.pop('colors', None) or kwargs.pop('pallette', 'jet'),
                            alpha=kwargs.pop('alpha', 1),
                            **kwargs)
        if ('kind' in kwargs and \
           kwargs['kind'] == 'map') or \
           (len(args) > 3 and args[3] == 'map') and \
           ('as_graphic' in kwargs and kwargs['as_graphic']):
            from arcgis.features import FeatureCollection, FeatureSet
            from arcgis import geometry
            if self._gis is None:
                gis = GIS(set_active=False)
            else:
                gis = self._gis
            if self.sr:
                sr = self.sr
            else:
                sr = self.sr
            extent = None
            if HASARCPY:
                if sr:
                    wkid = None
                    if hasattr(sr, 'factoryCode'):
                        wkid = {'wkid' : sr.factoryCode}
                    elif isinstance(sr, geometry.SpatialReference):
                        wkid = self.sr
                    ext = self.geoextent
                    extent = {
                        "xmin" : ext[0],
                        "ymin" : ext[1],
                        "xmax" : ext[2],
                        "ymax" : ext[3],
                        "spatialReference" : wkid
                    }
                else:
                    ext = self.geoextent
                    extent = {
                        "xmin" : ext[0],
                        "ymin" : ext[1],
                        "xmax" : ext[2],
                        "ymax" : ext[3],
                        "spatialReference" : {'wkid' : 4326}
                    }
            else:
                sr = self.sr
                if self.sr is None:
                    sr = {'wkid' : 4326}

                ext = self.geoextent
                extent = {
                    "xmin" : ext[0],
                    "ymin" : ext[1],
                    "xmax" : ext[2],
                    "ymax" : ext[3],
                    "spatialReference" : sr
                }
            if 'map_widget' not in kwargs:
                raise Exception("map_widget is required to plot the SpatialDataFrame")
            else:
                m = kwargs.pop('map_widget')
                symbol = kwargs.pop('symbol', None)
                popup = kwargs.pop('popup', None)
            try:
                fs = FeatureSet.from_dict(self.__feature_set__)
                m.draw(fs, symbol=symbol, popup=popup)
                if extent and \
                   isinstance(extent, dict):
                    m.extent = extent
            except:
                raise Exception('Could not plot the Spatial DataFrame.')
        else:
            return super(SpatialDataFrame, self).plot(*args, **kwargs)
    # ----------------------------------------------------------------------
    @staticmethod
    def from_xy(df, x_column, y_column, sr=4326):
        """
        Converts a Pandas DataFrame into a Spatial DataFrame by providing the X/Y columns.

        ====================    =========================================================
        **Argument**            **Description**
        --------------------    ---------------------------------------------------------
        df                      Required Pandas DataFrame. Source dataset
        --------------------    ---------------------------------------------------------
        x_column                Required string.  The name of the X-coordinate series
        --------------------    ---------------------------------------------------------
        y_column                Required string.  The name of the Y-coordinate series
        --------------------    ---------------------------------------------------------
        sr                      Optional int.  The wkid number of the spatial reference.
        ====================    =========================================================

        :returns: SpatialDataFrame

        """
        from .io.fileops import _from_xy
        return _from_xy(df=df, x_column=x_column,
                        y_column=y_column, sr=sr)
    # ----------------------------------------------------------------------
    @staticmethod
    def from_df(df, address_column="address", geocoder=None):
        """
        Returns a SpatialDataFrame from a dataframe with an address column.

        ====================    =========================================================
        **Argument**            **Description**
        --------------------    ---------------------------------------------------------
        df                      Required Pandas DataFrame. Source dataset
        --------------------    ---------------------------------------------------------
        address_column          Optional String. The default is "address". This is the
                                name of a column in the specified dataframe that contains
                                addresses (as strings). The addresses are batch geocoded
                                using the GIS's first configured geocoder and their
                                locations used as the geometry of the spatial dataframe.
                                Ignored if the 'geometry' parameter is also specified.
        --------------------    ---------------------------------------------------------
        geocoder                Optional Geocoder. The geocoder to be used. If not
                                specified, the active GIS's first geocoder is used.
        ====================    =========================================================

        :returns: SpatialDataFrame



        NOTE: Credits will be consumed for batch_geocoding, from
        the GIS to which the geocoder belongs.

        """
        from arcgis.geocoding import get_geocoders, geocode, batch_geocode
        if geocoder is None:
            geocoder = arcgis.env.active_gis._tools.geocoders[0]

        geoms = []
        if address_column in df.columns:
            # batch geocode addresses in the address column and use them as the geometry
            batch_size = geocoder.properties.locatorProperties.MaxBatchSize
            N = len(df)
            geoms = []
            for i in range(0, N, batch_size):
                start = i
                stop = i + batch_size if i + batch_size < N else N
                # print('Geocoding from ' + str(start) + ' to ' + str(stop))

                res = batch_geocode(list(df[start:stop][address_column]), geocoder=geocoder)
                for index in range(len(res)):
                    address = df.ix[start + index, address_column]
                    try:
                        loc = res[index]['location']
                        x = loc['x']
                        y = loc['y']
                        # self.ix[start + index, 'x'] = x
                        # self.ix[start + index, 'y'] = y
                        geoms.append(arcgis.geometry.Geometry({'x': x, 'y': y}))

                    except:
                        x, y = None, None
                        try:
                            loc = geocode(address, geocoder=geocoder)[0]['location']
                            x = loc['x']
                            y = loc['y']
                        except:
                            print('Unable to geocode address: ' + address)
                            pass
                        # self.ix[start + index, 'x'] = x
                        # self.ix[start + index, 'y'] = y
                        geoms.append(None)
        else:
            raise ValueError("Address column not found in dataframe")

        return SpatialDataFrame(df, geometry=geoms)
    #----------------------------------------------------------------------
    @staticmethod
    def from_featureclass(filename, **kwargs):
        """
        Returns a SpatialDataFrame from a feature class.

        ====================    =========================================================
        **Argument**            **Description**
        --------------------    ---------------------------------------------------------
        filename                Required string. The full path to the feature class
        --------------------    ---------------------------------------------------------
        sql_clause              Optional string. The sql clause to parse data down
        --------------------    ---------------------------------------------------------
        where_clause            Optional string. A where statement
        --------------------    ---------------------------------------------------------
        sr                      Optional SpatialReference. A spatial reference object
        ====================    =========================================================

        :returns: SpatialDataFrame

        """
        from .io import from_featureclass
        gis = kwargs.pop('gis', arcgis.env.active_gis)
        if HASARCPY:
            return from_featureclass(filename=filename, **kwargs)
        elif isinstance(gis, GIS) and \
             gis._con._auth.lower() != "anon":
            return from_featureclass(filename=filename, **kwargs)
        else:
            raise Exception("Cannot create the SpatialDataFrame, you must " +\
                            "have an authenticated GIS.")
    #----------------------------------------------------------------------
    @staticmethod
    def from_layer(layer, **kwargs):
        """
        Returns a SpatialDataFrame/Pandas' Dataframe from a FeatureLayer or Table object.

        ==============   ==============================================================
        **Arguments**    **Description**
        --------------   --------------------------------------------------------------
        layer            required FeatureLayer/Table. This is the service endpoint object.
        ==============   ==============================================================

        :returns: SpatialDataFrame for feature layers with geometry and Panda's Dataframe for tables

        """
        from .io import from_layer
        return from_layer(layer=layer, **kwargs)
    #----------------------------------------------------------------------
    def to_featureclass(self,
                        out_location, out_name,
                        overwrite=True, skip_invalid=True):
        """converts a SpatialDataFrame to a feature class

        ====================    =========================================================
        **Argument**            **Description**
        --------------------    ---------------------------------------------------------
        out_location            Required string. A save location workspace
        --------------------    ---------------------------------------------------------
        out_name                Required string. The name of the feature class to save as
        --------------------    ---------------------------------------------------------
        overwrite               Optional boolean. True means to erase and replace value,
                                false means to append
        --------------------    ---------------------------------------------------------
        skip_invalids           Optional boolean. If True, any bad rows will be ignored.
        ====================    =========================================================

        :returns: string

        """
        from .io import to_featureclass
        return to_featureclass(df=self,
                               out_location=out_location,
                               out_name=out_name,
                               overwrite=overwrite, skip_invalid=skip_invalid)
    #----------------------------------------------------------------------
    def to_hdf(self, path_or_buf, key, **kwargs):
        """Write the contained data to an HDF5 file using HDFStore.

        Parameters
        ----------
        path_or_buf : the path (string) or HDFStore object
        key : string
            indentifier for the group in the store
        mode : optional, {'a', 'w', 'r+'}, default 'a'

          ``'w'``
              Write; a new file is created (an existing file with the same
              name would be deleted).
          ``'a'``
              Append; an existing file is opened for reading and writing,
              and if the file does not exist it is created.
          ``'r+'``
              It is similar to ``'a'``, but the file must already exist.
        format : 'fixed(f)|table(t)', default is 'fixed'
            fixed(f) : Fixed format
                       Fast writing/reading. Not-appendable, nor searchable
            table(t) : Table format
                       Write as a PyTables Table structure which may perform
                       worse but allow more flexible operations like searching
                       / selecting subsets of the data
        append : boolean, default False
            For Table formats, append the input data to the existing
        data_columns :  list of columns, or True, default None
            List of columns to create as indexed data columns for on-disk
            queries, or True to use all columns. By default only the axes
            of the object are indexed. See `here
            <http://pandas.pydata.org/pandas-docs/stable/io.html#query-via-data-columns>`__.

            Applicable only to format='table'.
        complevel : int, 1-9, default 0
            If a complib is specified compression will be applied
            where possible
        complib : {'zlib', 'bzip2', 'lzo', 'blosc', None}, default None
            If complevel is > 0 apply compression to objects written
            in the store wherever possible
        fletcher32 : bool, default False
            If applying compression use the fletcher32 checksum
        dropna : boolean, default False.
            If true, ALL nan rows will not be written to store.
        """

        from pandas.io import pytables
        return pytables.to_hdf(path_or_buf, key, pd.DataFrame(self), **kwargs)
    #----------------------------------------------------------------------
    @staticmethod
    def from_hdf(path_or_buf, key=None, **kwargs):
        """ read from the store, close it if we opened it

            Retrieve pandas object stored in file, optionally based on where
            criteria

            Parameters
            ----------
            path_or_buf : path (string), buffer, or path object (pathlib.Path or
                py._path.local.LocalPath) to read from

                .. versionadded:: 0.19.0 support for pathlib, py.path.

            key : group identifier in the store. Can be omitted if the HDF file
                contains a single pandas object.
            where : list of Term (or convertable) objects, optional
            start : optional, integer (defaults to None), row number to start
                selection
            stop  : optional, integer (defaults to None), row number to stop
                selection
            columns : optional, a list of columns that if not None, will limit the
                return columns
            iterator : optional, boolean, return an iterator, default False
            chunksize : optional, nrows to include in iteration, return an iterator

            Returns
            -------
            The selected object

            """
        return SpatialDataFrame(pd.read_hdf(path_or_buf=path_or_buf,
                                            key=key, **kwargs))
    #----------------------------------------------------------------------
    def to_feature_collection(self,
                              name=None,
                              drawing_info=None,
                              extent=None,
                              global_id_field=None):
        """
        converts a Spatial DataFrame to a Feature Collection

        =====================  ===============================================================
        **optional argument**  **Description**
        ---------------------  ---------------------------------------------------------------
        name                   optional string. Name of the Feature Collection
        ---------------------  ---------------------------------------------------------------
        drawing_info           Optional dictionary. This is the rendering information for a
                               Feature Collection.  Rendering information is a dictionary with
                               the symbology, labelling and other properties defined.  See:
                               http://resources.arcgis.com/en/help/arcgis-rest-api/index.html#/Renderer_objects/02r30000019t000000/
        ---------------------  ---------------------------------------------------------------
        extent                 Optional dictionary.  If desired, a custom extent can be
                               provided to set where the map starts up when showing the data.
                               The default is the full extent of the dataset in the Spatial
                               DataFrame.
        ---------------------  ---------------------------------------------------------------
        global_id_field        Optional string. The Global ID field of the dataset.
        =====================  ===============================================================

        :returns: FeatureCollection object
        """
        from arcgis.features import FeatureCollection
        import uuid
        import string
        import random

        if name is None:
            name = random.choice(string.ascii_letters) + uuid.uuid4().hex[:5]
        template = {
            'showLegend' : True,
            'layers' : []
        }
        if extent is None:
            ext = self.geoextent
            extent = {
                "xmin" : ext[0],
                "ymin" : ext[1],
                "xmax" : ext[2],
                "ymax" : ext[3],
                "spatialReference" : self.sr
            }
        fs = self.__feature_set__
        fields = []
        for fld in fs['fields']:
            if fld['name'].lower() == fs['objectIdFieldName'].lower():
                fld['editable'] = False
                fld['sqlType'] = "sqlTypeOther"
                fld['domain'] = None
                fld['defaultValue'] = None
                fld['nullable'] = False
            else:
                fld['editable'] = True
                fld['sqlType'] = "sqlTypeOther"
                fld['domain'] = None
                fld['defaultValue'] = None
                fld['nullable'] = True
        if drawing_info is None:
            di = {
                'renderer' : {
                    'labelingInfo' : None,
                    'label' : "",
                    'description' : "",
                    'type' : 'simple',
                    'symbol' : None

                }
            }
            symbol = None
            if symbol is None:
                if fs['geometryType'] in ["esriGeometryPoint", "esriGeometryMultipoint"]:
                    di['renderer']['symbol'] = {"color":[0,128,0,128],"size":18,"angle":0,
                                                "xoffset":0,"yoffset":0,
                                                "type":"esriSMS",
                                                "style":"esriSMSCircle",
                                                "outline":{"color":[0,128,0,255],"width":1,
                                                           "type":"esriSLS","style":"esriSLSSolid"}}
                elif fs['geometryType'] == 'esriGeometryPolyline':
                    di['renderer']['symbol'] = {
                        "type": "esriSLS",
                        "style": "esriSLSDot",
                        "color": [0,128,0,128],
                        "width": 1
                    }
                elif fs['geometryType'] == 'esriGeometryPolygon':
                    di['renderer']['symbol'] = {
                        "type": "esriSFS",
                        "style": "esriSFSSolid",
                        "color": [0,128,0,128],
                        "outline": {
                            "type": "esriSLS",
                            "style": "esriSLSSolid",
                            "color": [110,110,110,255],
                            "width": 1
                        }
                    }
            else:
                di['renderer']['symbol'] = symbol
        else:
            di = drawing_info
        layer = {
            'featureSet' : {'features' : fs['features'],
                            'geometryType' : fs['geometryType']
                            },
            'layerDefinition' : {
                'htmlPopupType' : 'esriServerHTMLPopupTypeNone',
                'objectIdField' : fs['objectIdFieldName'] or "OBJECTID",
                #'types' : [],
                'defaultVisibility' : True,
                'supportsValidateSql' : True,
                'supportsAttachmentsByUploadId' : True,
                'useStandardizedQueries' : False,
                'supportsApplyEditsWithGlobalIds' : True,
                'standardMaxRecordCount' : 32000,
                'supportsTruncate' : False,
                'extent' : extent,
                'maxScale' : 0,
                'supportsAppend' : True,
                'supportsCalculate' : True,
                'copyrightText' : "",
                #'templates' : [],
                'description' : "",
                #'relationships' : [],
                'supportsRollbackOnFailureParameter' : True,
                'hasM' : False,
                'displayField' : "",
                'drawingInfo' : di,
                'type' : 'Feature Layer',
                'supportedQueryFormats' : 'JSON, geoJSON',
                'isDataVersioned' : False,
                'maxRecordCount' : 2000,
                'minScale' : 0,
                'supportsStatistics' : True,
                'hasAttachments' : False,
                #'indexes' : [],
                'tileMaxRecordCount' : 8000,
                'supportsAdvancedQueries' : True,
                #'globalIdField' : "",
                'hasZ' : False,
                'name' : name,
                'id' : 0,
                'allowGeometryUpdates' : True,
                #'typeIdField' : "",
                'geometryType' : fs['geometryType'],
                'currentVersion' : 10.51,
                #'maxRecordCountFactor' : 1,
                'supportsCoordinatesQuantization' : True,
                'fields' : fs['fields'],
                'hasStaticData' : True,# False
                'capabilities' : 'Create,Delete,Query,Update,Editing,Extract,Sync',
                'advancedQueryCapabilities' :  {'supportsReturningGeometryCentroid': False,
                                                'supportsQueryRelatedPagination': True,
                                                'supportsHavingClause': True,
                                                'supportsOrderBy': True,
                                                'supportsPaginationOnAggregatedQueries': True,
                                                'supportsQueryWithDatumTransformation': True,
                                                'supportsAdvancedQueryRelated': True,
                                                'supportsOutFieldSQLExpression': True,
                                                'supportsPagination': True,
                                                'supportsStatistics': True,
                                                'supportsSqlExpression': True,
                                                'supportsQueryWithDistance': True,
                                                'supportsReturningQueryExtent': True,
                                                'supportsDistinct': True,
                                                'supportsQueryWithResultType': True},

            }
        }
        if global_id_field is not None:
            layer['layerDefinition']['globalIdField'] = global_id_field
        return FeatureCollection(layer)
    #----------------------------------------------------------------------
    def to_featureset(self):
        """
        Converts a spatial dataframe to a feature set object
        """
        from arcgis.features import FeatureSet
        return FeatureSet.from_dataframe(self)
    #----------------------------------------------------------------------
    def _to_arcpy_featureset(self):
        """
        Converts a Spatial DataFrame to arcpy.FeatureSet so it can be used
        in geoprocessing tools.

        returns: arcpy.FeatureSet
        """
        if HAS_ARCPY:
            import uuid, string, random
            l = []
            for i in range(3):
                l.append(random.choice(string.ascii_letters))
            l = "".join(l)
            out_name = l
            res = self.to_featureclass(out_location='in_memory',
                                       out_name=out_name)

            feature_set = arcpy.FeatureSet()
            feature_set.load(res)
            return feature_set
        else:
            raise Exception("ArcPy must be present to convert to arcpy.FeatureSet object")
    #----------------------------------------------------------------------
    def to_featurelayer(self,
                        title,
                        gis=None,
                        tags=None):
        """
        publishes a spatial dataframe to a new feature layer

        ===========================     ====================================================================
        **Argument**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        title                           Required string. The name of the service
        ---------------------------     --------------------------------------------------------------------
        gis                             Optional GIS. The GIS connection object
        ---------------------------     --------------------------------------------------------------------
        tags                            Optional string. A comma seperated list of descriptive words for the
                                        service
        ===========================     ====================================================================

        :returns: FeatureLayer

        """
        from arcgis import env
        if gis is None:
            gis = env.active_gis
            if gis is None:
                raise ValueError("GIS object must be provided")
        content = gis.content
        return content.import_data(self, title=title, tags=tags)
    #----------------------------------------------------------------------
    def set_geometry(self, col, drop=False, inplace=False, sr=None):
        """
        Set the SpatialDataFrame geometry using either an existing column or
        the specified input. By default yields a new object.

        The original geometry column is replaced with the input.

        ===========================     ====================================================================
        **Argument**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        col                             Required string/np.array. column label or array
        ---------------------------     --------------------------------------------------------------------
        drop                            Optional boolean. Default True. Delete column to be used as the new
                                        geometry
        ---------------------------     --------------------------------------------------------------------
        inplace                         Optional boolean. Default False. Modify the SpatialDataFrame in
                                        place (do not create a new object)
        ---------------------------     --------------------------------------------------------------------
        sr                              Optional SpatialReference/Integer. The wkid value Coordinate
                                        system to use. If passed, overrides both DataFrame and col's sr.
                                        Otherwise, tries to get sr from passed col values or
                                        DataFrame.
        ===========================     ====================================================================

        :returns: SpatialDataFrame

        """
        if inplace:
            frame = self
        else:
            frame = self.copy()
        if sr:
            sr = self._sr(sr=sr)
        if not sr:
            sr = getattr(col, 'sr', None)
            if sr is None and \
               isinstance(col, GeoSeries):
                col.sr = self.sr
        to_remove = None
        if isinstance(col, string_types):
            geo_column_name = col
            self._geometry_column_name = col
        else:
            geo_column_name = self._geometry_column_name
        if isinstance(col, (GeoSeries, Series, list, numpy.ndarray)):
            level = col
        elif hasattr(col, 'ndim') and col.ndim != 1:
            raise ValueError("Must pass array with one dimension only.")
        else:
            try:
                level = frame[col].values
            except KeyError:
                raise ValueError("Unknown column %s" % col)
            except:
                raise
            if drop:
                to_remove = col
                geo_column_name = self._geometry_column_name
            else:
                geo_column_name = col

        if to_remove:
            del frame[to_remove]

        if isinstance(level, GeoSeries) and level.sr != sr:
            # Avoids caching issues/sr sharing issues
            level = level.copy()
            level.sr = sr
        # Check that we are using a listlike of geometries
        if not all(isinstance(item, GEOM_TYPES) or not item for item in level):
            raise TypeError("Input geometry column must contain valid geometry objects.")
        #if isinstance(frame[geo_column_name], pd.Series):
        #    frame[geo_column_name] = GeoSeries(frame[geo_column_name])
        if isinstance(level, (list, tuple, numpy.ndarray)):
            level = GeoSeries(level)
        frame[geo_column_name] = level
        frame._geometry_column_name = geo_column_name
        frame.sr = sr
        frame._delete_index()

        if (frame.sr != self.sr and HASARCPY):
            if isinstance(sr, dict):
                if hasattr(sr, 'as_arcpy') and HASARCPY:
                    sr = sr.as_arcpy
                elif 'wkid' in sr:
                    sr = sr['wkid']
                elif 'wkt' in sr:
                    sr = sr['wkt']
        import json
        gtypes = frame.geometry.apply(lambda x: type(x)).unique()
        if len(gtypes) == 1 and \
           gtypes[0] in [_types.Point, _types.Polygon, _types.Polyline]:
            pass
        elif HASARCPY: # Use ArcPy to Enforce Proper Geometry Construction
            for idx, g in frame.geometry.iteritems():
                if isinstance(g, arcpy.Point):
                    g = arcgis.geometry.Geometry(json.loads(arcpy.PointGeometry(g, sr).JSON))
                elif hasattr(g, "JSON"):
                    g = arcgis.geometry.Geometry(json.loads(g.JSON))
                elif isinstance(g, string_types):
                    g = arcgis.geometry.Geometry(json.loads(g))
                elif isinstance(g, dict):
                    g = arcgis.geometry.Geometry(g)

                if inplace:
                    try:
                        frame.loc[idx, self._geometry_column_name] = g
                    except:
                        try:
                            frame.at[idx, self._geometry_column_name] = g
                        except:
                            frame.set_value(index=idx,
                                        col=self._geometry_column_name,
                                        value=g)
                else:
                    try:
                        frame.iloc[idx, self._geometry_column_name] = g
                    except:
                        frame.loc[idx, self._geometry_column_name] = g
                del idx, g
            if sr:
                frame.sr = self._sr(sr)
                frame.geometry = frame.geometry.project_as(sr)
        else:
            sr = self.sr
            if sr is None:
                sr = {'wkid' : 4326}
            for idx, g in frame.geometry.iteritems():
                if hasattr(g, "JSON") and HASARCPY:
                    g = arcgis.geometry.Geometry(json.loads(g.JSON))
                elif str(type(g)) == "<class 'arcpy.arcobjects.arcobjects.Point'>":
                    g = arcgis.geometry.Geometry({'x':g.X,
                                                  'y':g.Y,
                                                  'spatialReference':sr})
                elif isinstance(g, string_types):
                    g = arcgis.geometry.Geometry(json.loads(g))
                elif isinstance(g, dict):
                    g = arcgis.geometry.Geometry(g)
                else:
                    raise ValueError("Invalid Geometry")
                if sr is None:
                    sr = {'wkid' : 4326}
                if 'spatialReference' not in g:
                    g['spatialReference'] = dict(sr)
                if inplace:
                    try:
                        frame.loc[idx, self._geometry_column_name] = g
                    except:
                        frame.set_value(index=idx, col=self._geometry_column_name, value=g)
                else:
                    try:
                        frame.iloc[idx, self._geometry_column_name] = g
                    except:
                        frame.set_value(index=idx, col=self._geometry_column_name, value=g)
            frame.sr = self._sr(sr)
        if not inplace:
            return frame
        self = frame
    #----------------------------------------------------------------------
    def __getitem__(self, key):
        """
        If the result is a column containing only 'geometry', return a
        GeoSeries. If it's a DataFrame with a 'geometry' column, return a
        SpatialDataFrame.
        """
        result = super(SpatialDataFrame, self).__getitem__(key)
        geo_col = self._geometry_column_name
        if isinstance(key, string_types) and key == geo_col:
            result.__class__ = GeoSeries
            result.sr = self.sr
            result._delete_index()
        elif isinstance(result, DataFrame) and geo_col in result:
            result.__class__ = SpatialDataFrame
            result.sr = self.sr
            result._geometry_column_name = geo_col
            result._delete_index()
        elif isinstance(result, DataFrame) and geo_col not in result:
            result.__class__ = DataFrame
        return result
    #----------------------------------------------------------------------
    def reproject(self, spatial_reference, transformation=None, inplace=False):
        """
        Reprojects a given dataframe into a new coordinate system.

        ===========================     ====================================================================
        **Argument**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        spatial_reference               Required Integer/SpatialReference. The spatial reference the data
                                        should be reprojected into.
        ---------------------------     --------------------------------------------------------------------
        transformation                  Optional string. The optional transformation string.
        ---------------------------     --------------------------------------------------------------------
        inplace                         Optional boolean. Default False. Modify the SpatialDataFrame in
                                        place (do not create a new object)
        ===========================     ====================================================================

        :returns: SpatialDataFrame

        """
        if HASARCPY:
            if isinstance(spatial_reference, arcpy.SpatialReference):
                wkt = spatial_reference.exportToString()
                wkid = spatial_reference.factoryCode
                if wkid:
                    sr = _types.SpatialReference({'wkid' : wkid})
                elif wkt:
                    sr = _types.SpatialReference({'wkt': wkt})
                else:
                    sr = None
            elif isinstance(spatial_reference, int):
                sr = _types.SpatialReference({'wkid' : spatial_reference})
            elif isinstance(spatial_reference, string_types):
                sr = _types.SpatialReference({'wkt' : spatial_reference})
            elif isinstance(spatial_reference, _types.SpatialReference):
                sr = spatial_reference
            else:
                raise ValueError("spatial_referernce must be of type: int, string, _types.SpatialReference, or arcpy.SpatialReference")

            if inplace:
                df = self
            else:
                df = self.copy()
            sarcpy = sr.as_arcpy
            if sarcpy:
                geom = df.geometry.project_as(sarcpy, transformation)
                geom.sr = sr
                df.geometry = geom
                if inplace:
                    return df
            else:
                raise Exception("could not reproject the dataframe.")
            return df
    #----------------------------------------------------------------------
    def select_by_location(self, other, matches_only=True):
        """
        Selects all rows in a given SpatialDataFrame based on a given geometry

        ===========================     ====================================================================
        **Argument**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        other                           Required Geometry. A geometry object to check for intersection.
        ---------------------------     --------------------------------------------------------------------
        matches_only                    Optional boolean.  if true, only matched records will be
                                        returned, else a field called 'select_by_location' will be added
                                        to the dataframe with the results of the select by location.
        ===========================     ====================================================================

        :returns: SpatialDataFrame

        """
        if isinstance(other, Geometry):
            if self.geometry_type.lower() == 'point':
                res = self.within(other)
            else:
                res = self.overlaps(other)
            if matches_only:
                return self[res]
            else:
                self['select_by_location'] = res
        else:
            raise ValueError("Input must be a geometry")
    #----------------------------------------------------------------------
    def merge_datasets(self, other):
        """
        This operation combines two dataframes into one new DataFrame.
        If the operation is combining two SpatialDataFrames, the
        geometry_type must match.

        ===========================     ====================================================================
        **Argument**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        other                           Required SpatialDataFrame. Another SpatialDataFrame to combine.
        ===========================     ====================================================================

        :returns: SpatialDataFrame

        """
        if isinstance(other, SpatialDataFrame) and \
           other.geometry_type == self.geometry_type:
            return pd.concat(objs=[self, other], axis=0)
        elif isinstance(other, DataFrame):
            return pd.concat(objs=[self, other], axis=0)
        elif isinstance(other, Series):
            self['merged_datasets'] = other
        elif isinstance(other, SpatialDataFrame) and \
             other.geometry_type != self.geometry_type:
            raise ValueError("Spatial DataFrames must have the same geometry type.")
        else:
            raise ValueError("Merge datasets cannot merge types %s" % type(other))
    #----------------------------------------------------------------------
    def erase(self, other, inplace=False):
        """
        Erases

        ===========================     ====================================================================
        **Argument**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        other                           Required Geometry. A geometry object to erase from other geometries.
        ---------------------------     --------------------------------------------------------------------
        inplace                         Optional boolean. Default False. Modify the SpatialDataFrame in
                                        place (do not create a new object)
        ===========================     ====================================================================

        :returns: SpatialDataFrame
        """
        if inplace:
            df = self
        else:
            df = self.copy()
        if isinstance(other, Geometry):
            df.geometry = self.geometry.symmetricDifference(other)
            return df
        else:
            raise ValueError("Input must be of type arcpy.Geometry, not %s" % type(other))

###########################################################################
def _dataframe_set_geometry(self, col, drop=False, inplace=False, sr=None):
    if inplace:
        raise ValueError("Can't do inplace setting when converting from"
                         " DataFrame to SpatialDataFrame")
    gf = SpatialDataFrame(self)
    # this will copy so that BlockManager gets copied
    return gf.set_geometry(col, drop=drop, inplace=False, sr=sr)

if PY3:
    DataFrame.set_geometry = _dataframe_set_geometry
else:
    import types
    DataFrame.set_geometry = types.MethodType(_dataframe_set_geometry, None,
                                              DataFrame)
