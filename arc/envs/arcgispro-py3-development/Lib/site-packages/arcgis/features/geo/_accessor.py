"""
Holds Delegate and Accessor Logic
"""
import os
import copy
import uuid
import shutil
import logging
import datetime
import tempfile
import pandas as pd
import numpy as np
from collections.abc import Iterable
from ._internals import register_dataframe_accessor, register_series_accessor
from ._array import GeoType
from ._io.fileops import to_featureclass, from_featureclass, _sanitize_column_names, read_feather
from arcgis.geometry import Geometry, SpatialReference, Envelope, Point
from arcgis._impl.common._mixins import PropertyMap
from arcgis._impl.common._isd import InsensitiveDict
_LOGGER = logging.getLogger(__name__)
############################################################################
def _is_geoenabled(df):
    """
    Checks if a Panda's DataFrame is 'geo-enabled'.

    This means that a spatial column is defined and is a GeoArray

    :returns: boolean
    """
    try:
        if isinstance(df, pd.DataFrame) and \
           hasattr(df, 'spatial') and \
           df.spatial.name and \
           df[df.spatial.name].dtype.name.lower() == 'geometry':
            return True
        else:
            return False
    except:
        return False
###########################################################################
@pd.api.extensions.register_series_accessor("geom")
class GeoSeriesAccessor:
    """
    """
    _data = None
    _index = None
    _name = None
    #----------------------------------------------------------------------
    def __init__(self, obj):
        """initializer"""
        self._validate(obj)
        self._data = obj.values
        self._index = obj.index
        self._name = obj.name
    #----------------------------------------------------------------------
    @staticmethod
    def _validate(obj):
        if not is_geometry_type(obj):
            raise AttributeError("Cannot use 'geom' accessor on objects of "
                                 "dtype '{}'.".format(obj.dtype))
    ##---------------------------------------------------------------------
    ##   Accessor Properties
    ##---------------------------------------------------------------------
    @property
    def area(self):
        """
        Returns the features area

        :returns: float in a series
        """
        return pd.Series(self._data.area, name='area', index=self._index)

    #----------------------------------------------------------------------
    @property
    def as_arcpy(self):
        """
        Returns the features as ArcPy Geometry

        :returns: arcpy.Geometry in a series
        """
        return pd.Series(self._data.as_arcpy, name='as_arcpy', index=self._index)
    #----------------------------------------------------------------------
    @property
    def as_shapely(self):
        """
        Returns the features as Shapely Geometry

        :returns: shapely.Geometry in a series
        """
        return pd.Series(self._data.as_shapely, name='as_shapely', index=self._index)
    #----------------------------------------------------------------------
    @property
    def centroid(self):
        """
        Returns the feature's centroid

        :returns: tuple (x,y) in series
        """
        return pd.Series(self._data.centroid, name='centroid', index=self._index)
    #----------------------------------------------------------------------
    @property
    def extent(self):
        """
        Returns the feature's extent

        :returns: tuple (xmin,ymin,xmax,ymax) in series
        """
        return pd.Series(self._data.extent, name='extent', index=self._index)
    #----------------------------------------------------------------------
    @property
    def first_point(self):
        """
        Returns the feature's first point

        :returns: Geometry
        """
        return pd.Series(self._data.first_point, name='first_point', index=self._index)
    #----------------------------------------------------------------------
    @property
    def geoextent(self):
        """
        A returns the geometry's extents

        :returns: Series of Floats
        """
        #res = self._data.geoextent
        #res.index = self._index
        return pd.Series(self._data.geoextent, name='geoextent', index=self._index)
    #----------------------------------------------------------------------
    @property
    def geometry_type(self):
        """
        returns the geometry types

        :returns: Series of strings
        """
        return pd.Series(self._data.geometry_type, name='geometry_type', index=self._index)
    #----------------------------------------------------------------------
    @property
    def hull_rectangle(self):
        """
        A space-delimited string of the coordinate pairs of the convex hull

        :returns: Series of strings
        """
        return pd.Series(self._data.hull_rectangle, name='hull_rectangle', index=self._index)
    #----------------------------------------------------------------------
    @property
    def has_z(self):
        """
        Determines if the geometry has a Z value

        :returns: Series of Boolean
        """
        return pd.Series(self._data.has_z, name='has_z', index=self._index)
    #----------------------------------------------------------------------
    @property
    def has_m(self):
        """
        Determines if the geometry has a M value

        :returns: Series of Boolean
        """
        return pd.Series(self._data.has_m, name='has_m', index=self._index)
    #----------------------------------------------------------------------
    @property
    def is_empty(self):
        """
        Returns True/False if feature is empty

        :returns: Series of Booleans
        """
        return pd.Series(self._data.is_empty, name='is_empty', index=self._index)
    #----------------------------------------------------------------------
    @property
    def is_multipart(self):
        """
        Returns True/False if features has multiple parts

        :returns: Series of Booleans
        """
        return pd.Series(self._data.is_multipart, name='is_multipart', index=self._index)
    #----------------------------------------------------------------------
    @property
    def is_valid(self):
        """
        Returns True/False if features geometry is valid

        :returns: Series of Booleans
        """
        return pd.Series(self._data.is_valid, name='is_valid', index=self._index)
    #----------------------------------------------------------------------
    @property
    def JSON(self):
        """
        Returns JSON string  of Geometry

        :returns: Series of strings
        """
        return pd.Series(self._data.JSON, name='JSON', index=self._index)
    #----------------------------------------------------------------------
    @property
    def label_point(self):
        """
        Returns the geometry point for the optimal label location

        :returns: Series of Geometries
        """
        return pd.Series(self._data.label_point, name='label_point', index=self._index)
    #----------------------------------------------------------------------
    @property
    def last_point(self):
        """
        Returns the Geometry of the last point in a feature.

        :returns: Series of Geometry
        """
        return pd.Series(self._data.last_point, name='last_point', index=self._index)
    #----------------------------------------------------------------------
    @property
    def length(self):
        """
        Returns the length of the features

        :returns: Series of float
        """
        return pd.Series(self._data.length, name='length', index=self._index)
    #----------------------------------------------------------------------
    @property
    def length3D(self):
        """
        Returns the length of the features

        :returns: Series of float
        """
        return pd.Series(self._data.length3D, name='length3D', index=self._index)
    #----------------------------------------------------------------------
    @property
    def part_count(self):
        """
        Returns the number of parts in a feature's geometry

        :returns: Series of Integer
        """
        return pd.Series(self._data.part_count, name='part_count', index=self._index)
    #----------------------------------------------------------------------
    @property
    def point_count(self):
        """
        Returns the number of points in a feature's geometry

        :returns: Series of Integer
        """
        return pd.Series(self._data.part_count, name='point_count', index=self._index)
    #----------------------------------------------------------------------
    @property
    def spatial_reference(self):
        """
        Returns the Spatial Reference of the Geometry

        :returns: Series of SpatialReference
        """
        return pd.Series(self._data.spatial_reference, name='spatial_reference', index=self._index)
    #----------------------------------------------------------------------
    @property
    def true_centroid(self):
        """
        Returns the true centroid of the Geometry

        :returns: Series of Points
        """
        return pd.Series(self._data.true_centroid, name='true_centroid', index=self._index)
    #----------------------------------------------------------------------
    @property
    def WKB(self):
        """
        Returns the Geometry as WKB

        :returns: Series of Bytes
        """
        return pd.Series(self._data.WKB, name='WKB', index=self._index)
    #----------------------------------------------------------------------
    @property
    def WKT(self):
        """
        Returns the Geometry as WKT

        :returns: Series of String
        """
        return pd.Series(self._data.WKT, name='WKT', index=self._index)
    ##---------------------------------------------------------------------
    ##  Accessor Geometry Method
    ##---------------------------------------------------------------------
    def angle_distance_to(self, second_geometry, method="GEODESIC"):
        """
        Returns a tuple of angle and distance to another point using a
        measurement type.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required Geometry.  A arcgis.Geometry object.
        ---------------     --------------------------------------------------------------------
        method              Optional String. PLANAR measurements reflect the projection of geographic
                            data onto the 2D surface (in other words, they will not take into
                            account the curvature of the earth). GEODESIC, GREAT_ELLIPTIC,
                            LOXODROME, and PRESERVE_SHAPE measurement types may be chosen as
                            an alternative, if desired.
        ===============     ====================================================================

        :returns: a tuple of angle and distance to another point using a measurement type.
        """
        res = self._data.angle_distance_to(**{'second_geometry' : second_geometry,
                                               'method' : method})
        return pd.Series(res, index=self._index, name='angle_distance_to')
    #----------------------------------------------------------------------
    def boundary(self):
        """
        Constructs the boundary of the geometry.

        :returns: arcgis.geometry.Polyline
        """
        return pd.Series(self._data.boundary(),
                         index=self._index,
                         name='boundary')
    #----------------------------------------------------------------------
    def buffer(self, distance):
        """
        Constructs a polygon at a specified distance from the geometry.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        distance            Required float. The buffer distance. The buffer distance is in the
                            same units as the geometry that is being buffered.
                            A negative distance can only be specified against a polygon geometry.
        ===============     ====================================================================

        :returns: arcgis.geometry.Polygon
        """
        return pd.Series(self._data.buffer(**{'distance' : distance}),
                         index=self._index,
                         name='buffer')
    #----------------------------------------------------------------------
    def clip(self, envelope):
        """
        Constructs the intersection of the geometry and the specified extent.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        envelope            required tuple. The tuple must have (XMin, YMin, XMax, YMax) each value
                            represents the lower left bound and upper right bound of the extent.
        ===============     ====================================================================

        :returns: output geometry clipped to extent

        """
        return pd.Series(self._data.clip(**{'envelope' : envelope}),
                         index=self._index,
                         name='clip')
    #----------------------------------------------------------------------
    def contains(self, second_geometry, relation=None):
        """
        Indicates if the base geometry contains the comparison geometry.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ---------------     --------------------------------------------------------------------
        relation            Optional string. The spatial relationship type.

                            + BOUNDARY - Relationship has no restrictions for interiors or boundaries.
                            + CLEMENTINI - Interiors of geometries must intersect. Specifying CLEMENTINI is equivalent to specifying None. This is the default.
                            + PROPER - Boundaries of geometries must not intersect.
        ===============     ====================================================================

        :returns: boolean
        """
        return pd.Series(self._data.contains(**{'second_geometry' : second_geometry,
                                                'relation' : relation}),
                         name='contains',
                         index=self._index)

    #----------------------------------------------------------------------
    def convex_hull(self):
        """
        Constructs the geometry that is the minimal bounding polygon such
        that all outer angles are convex.
        """
        return pd.Series(self._data.convex_hull(),
                         index=self._index,
                         name='convex_hull')
    #----------------------------------------------------------------------
    def crosses(self, second_geometry):
        """
        Indicates if the two geometries intersect in a geometry of a lesser
        shape type.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :returns: boolean

        """
        return pd.Series(self._data.crosses(**{'second_geometry' : second_geometry}),
                         name='crosses',
                         index=self._index)
    #----------------------------------------------------------------------
    def cut(self, cutter):
        """
        Splits this geometry into a part left of the cutting polyline, and
        a part right of it.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        cutter              Required Polyline. The cuttin polyline geometry
        ===============     ====================================================================

        :returns: a list of two geometries

        """
        return pd.Series(self._data.cut(**{'cutter' : cutter}),
                         index=self._index,
                         name='cut')
    #----------------------------------------------------------------------
    def densify(self, method, distance, deviation):
        """
        Creates a new geometry with added vertices

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        method              Required String. The type of densification, DISTANCE, ANGLE, or GEODESIC
        ---------------     --------------------------------------------------------------------
        distance            Required float. The maximum distance between vertices. The actual
                            distance between vertices will usually be less than the maximum
                            distance as new vertices will be evenly distributed along the
                            original segment. If using a type of DISTANCE or ANGLE, the
                            distance is measured in the units of the geometry's spatial
                            reference. If using a type of GEODESIC, the distance is measured
                            in meters.
        ---------------     --------------------------------------------------------------------
        deviation           Required float. Densify uses straight lines to approximate curves.
                            You use deviation to control the accuracy of this approximation.
                            The deviation is the maximum distance between the new segment and
                            the original curve. The smaller its value, the more segments will
                            be required to approximate the curve.
        ===============     ====================================================================

        :returns: arcgis.geometry.Geometry

        """
        return pd.Series(self._data.densify(**{'method' : method,
                                               'distance' : distance,
                                               'deviation' : deviation}),
                         index=self._index,
                         name='densify')
    #----------------------------------------------------------------------
    def difference(self, second_geometry):
        """
        Constructs the geometry that is composed only of the region unique
        to the base geometry but not part of the other geometry. The
        following illustration shows the results when the red polygon is the
        source geometry.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :returns: arcgis.geometry.Geometry

        """
        return pd.Series(self._data.difference(**{'second_geometry' : second_geometry}),
                         index=self._index,
                         name='difference')
    #----------------------------------------------------------------------
    def disjoint(self, second_geometry):
        """
        Indicates if the base and comparison geometries share no points in
        common.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :returns: boolean

        """
        res = self._data.disjoint(**{'second_geometry' : second_geometry})
        return pd.Series(res, index=self._index, name='disjoint')
    #----------------------------------------------------------------------
    def distance_to(self, second_geometry):
        """
        Returns the minimum distance between two geometries. If the
        geometries intersect, the minimum distance is 0.
        Both geometries must have the same projection.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :returns: float

        """
        res = self._data.distance_to(**{'second_geometry' : second_geometry})
        return pd.Series(res,
                         index=self._index,
                         name='distance_to')
    #----------------------------------------------------------------------
    def equals(self, second_geometry):
        """
        Indicates if the base and comparison geometries are of the same
        shape type and define the same set of points in the plane. This is
        a 2D comparison only; M and Z values are ignored.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :returns: boolean


        """
        return pd.Series(self._data.equals(**{'second_geometry' : second_geometry}),
                         name='equals',
                         index=self._index)
    #----------------------------------------------------------------------
    def generalize(self, max_offset):
        """
        Creates a new simplified geometry using a specified maximum offset
        tolerance.  This only works on Polylines and Polygons.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        max_offset          Required float. The maximum offset tolerance.
        ===============     ====================================================================

        :returns: arcgis.geometry.Geometry

        """
        res = self._data.generalize(**{'max_offset' : max_offset})
        return pd.Series(res,
                         index=self._index,
                         name='generalize')
    #----------------------------------------------------------------------
    def get_area(self, method, units=None):
        """
        Returns the area of the feature using a measurement type.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        method              Required String. PLANAR measurements reflect the projection of
                            geographic data onto the 2D surface (in other words, they will not
                            take into account the curvature of the earth). GEODESIC,
                            GREAT_ELLIPTIC, LOXODROME, and PRESERVE_SHAPE measurement types
                            may be chosen as an alternative, if desired.
        ---------------     --------------------------------------------------------------------
        units               Optional String. Areal unit of measure keywords: ACRES | ARES | HECTARES
                            | SQUARECENTIMETERS | SQUAREDECIMETERS | SQUAREINCHES | SQUAREFEET
                            | SQUAREKILOMETERS | SQUAREMETERS | SQUAREMILES |
                            SQUAREMILLIMETERS | SQUAREYARDS
        ===============     ====================================================================

        :returns: float

        """
        res = self._data.get_area(**{'method' : method,
                                      'units' : units})
        return pd.Series(res,
                         index=self._index,
                         name='get_area')
    #----------------------------------------------------------------------
    def get_length(self, method, units):
        """
        Returns the length of the feature using a measurement type.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        method              Required String. PLANAR measurements reflect the projection of
                            geographic data onto the 2D surface (in other words, they will not
                            take into account the curvature of the earth). GEODESIC,
                            GREAT_ELLIPTIC, LOXODROME, and PRESERVE_SHAPE measurement types
                            may be chosen as an alternative, if desired.
        ---------------     --------------------------------------------------------------------
        units               Required String. Linear unit of measure keywords: CENTIMETERS |
                            DECIMETERS | FEET | INCHES | KILOMETERS | METERS | MILES |
                            MILLIMETERS | NAUTICALMILES | YARDS
        ===============     ====================================================================

        :returns: float

        """
        res = self._data.get_length(**{'method' : method,
                                       'units' : units})
        return pd.Series(res,
                         index=self._index,
                         name='get_length')
    #----------------------------------------------------------------------
    def get_part(self, index=None):
        """
        Returns an array of point objects for a particular part of geometry
        or an array containing a number of arrays, one for each part.

        **requires arcpy**

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        index               Required Integer. The index position of the geometry.
        ===============     ====================================================================

        :return: arcpy.Array

        """
        res = self._data.get_part(**{'index' : index})
        return pd.Series(res,
                         index=self._index,
                         name='get_part')
    #----------------------------------------------------------------------
    def intersect(self, second_geometry, dimension=1):
        """
        Constructs a geometry that is the geometric intersection of the two
        input geometries. Different dimension values can be used to create
        different shape types. The intersection of two geometries of the
        same shape type is a geometry containing only the regions of overlap
        between the original geometries.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ---------------     --------------------------------------------------------------------
        dimension           Required Integer. The topological dimension (shape type) of the
                            resulting geometry.

                            + 1  -A zero-dimensional geometry (point or multipoint).
                            + 2  -A one-dimensional geometry (polyline).
                            + 4  -A two-dimensional geometry (polygon).

        ===============     ====================================================================

        :returns: boolean

        """
        return pd.Series(self._data.intersect(**{'second_geometry' : second_geometry,
                                                'dimension' : dimension}),
                         name='intersect',
                         index=self._index)
    #----------------------------------------------------------------------
    def measure_on_line(self, second_geometry, as_percentage=False):
        """
        Returns a measure from the start point of this line to the in_point.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ---------------     --------------------------------------------------------------------
        as_percentage       Optional Boolean. If False, the measure will be returned as a
                            distance; if True, the measure will be returned as a percentage.
        ===============     ====================================================================

        :return: float

        """
        res = self._data.measure_on_line(**{'second_geometry' : second_geometry,
                                            'as_percentage' : as_percentage})
        return pd.Series(res,
                         index=self._index,
                         name='measure_on_line')
    #----------------------------------------------------------------------
    def overlaps(self, second_geometry):
        """
        Indicates if the intersection of the two geometries has the same
        shape type as one of the input geometries and is not equivalent to
        either of the input geometries.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :return: boolean

        """
        return pd.Series(self._data.overlaps(**{'second_geometry' : second_geometry}),
                         name='overlaps',
                         index=self._index)
    #----------------------------------------------------------------------
    def point_from_angle_and_distance(self, angle, distance, method='GEODESCIC'):
        """
        Returns a point at a given angle and distance in degrees and meters
        using the specified measurement type.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        angle               Required Float. The angle in degrees to the returned point.
        ---------------     --------------------------------------------------------------------
        distance            Required Float. The distance in meters to the returned point.
        ---------------     --------------------------------------------------------------------
        method              Optional String. PLANAR measurements reflect the projection of geographic
                            data onto the 2D surface (in other words, they will not take into
                            account the curvature of the earth). GEODESIC, GREAT_ELLIPTIC,
                            LOXODROME, and PRESERVE_SHAPE measurement types may be chosen as
                            an alternative, if desired.
        ===============     ====================================================================

        :return: arcgis.geometry.Geometry


        """
        res = self._data.point_from_angle_and_distance(**{'angle' : angle,
                                                           'distance' : distance,
                                                           'method' : method})
        return pd.Series(res,
                         index=self._index,
                         name='point_from_angle_and_distance')
    #----------------------------------------------------------------------
    def position_along_line(self, value, use_percentage=False):
        """
        Returns a point on a line at a specified distance from the beginning
        of the line.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        value               Required Float. The distance along the line.
        ---------------     --------------------------------------------------------------------
        use_percentage      Optional Boolean. The distance may be specified as a fixed unit
                            of measure or a ratio of the length of the line. If True, value
                            is used as a percentage; if False, value is used as a distance.
                            For percentages, the value should be expressed as a double from
                            0.0 (0%) to 1.0 (100%).
        ===============     ====================================================================

        :return: Geometry

        """
        res = self._data.position_along_line(**{'value' : value,
                                                'use_percentage' : use_percentage})
        return pd.Series(res,
                         index=self._index,
                         name='position_along_line')
    #----------------------------------------------------------------------
    def project_as(self, spatial_reference, transformation_name=None):
        """
        Projects a geometry and optionally applies a geotransformation.

        ====================     ====================================================================
        **Argument**             **Description**
        --------------------     --------------------------------------------------------------------
        spatial_reference        Required SpatialReference. The new spatial reference. This can be a
                                 SpatialReference object or the coordinate system name.
        --------------------     --------------------------------------------------------------------
        transformation_name      Required String. The geotransformation name.
        ====================     ====================================================================

        :returns: arcgis.geometry.Geometry
        """
        res = self._data.project_as(**{'spatial_reference' : spatial_reference,
                                       'transformation_name' : transformation_name})
        return pd.Series(res,
                         index=self._index,
                         name='project_as')
    #----------------------------------------------------------------------
    def query_point_and_distance(self, second_geometry,
                                 use_percentage=False):
        """
        Finds the point on the polyline nearest to the in_point and the
        distance between those points. Also returns information about the
        side of the line the in_point is on as well as the distance along
        the line where the nearest point occurs.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ---------------     --------------------------------------------------------------------
        as_percentage       Optional boolean - if False, the measure will be returned as
                            distance, True, measure will be a percentage
        ===============     ====================================================================

        :return: tuple

        """
        res = self._data.query_point_and_distance(**{'second_geometry' : second_geometry,
                                                      'use_percentage' : use_percentage})
        return pd.Series(res,
                         index=self._index,
                         name='query_point_and_distance')
    #----------------------------------------------------------------------
    def segment_along_line(self, start_measure,
                           end_measure, use_percentage=False):
        """
        Returns a Polyline between start and end measures. Similar to
        Polyline.positionAlongLine but will return a polyline segment between
        two points on the polyline instead of a single point.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        start_measure       Required Float. The starting distance from the beginning of the line.
        ---------------     --------------------------------------------------------------------
        end_measure         Required Float. The ending distance from the beginning of the line.
        ---------------     --------------------------------------------------------------------
        use_percentage      Optional Boolean. The start and end measures may be specified as
                            fixed units or as a ratio.
                            If True, start_measure and end_measure are used as a percentage; if
                            False, start_measure and end_measure are used as a distance. For
                            percentages, the measures should be expressed as a double from 0.0
                            (0 percent) to 1.0 (100 percent).
        ===============     ====================================================================

        :returns: Geometry

        """
        res = self._data.segment_along_line(**{'start_measure' : start_measure,
                                               'end_measure' : end_measure,
                                               'use_percentage' : use_percentage})
        return pd.Series(res,
                         index=self._index,
                         name='segment_along_line')
    #----------------------------------------------------------------------
    def snap_to_line(self, second_geometry):
        """
        Returns a new point based on in_point snapped to this geometry.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :return: arcgis.gis.Geometry

        """
        res = self._data.snap_to_line(**{'second_geometry' : second_geometry})
        return pd.Series(res,
                         index=self._index,
                         name='snap_to_line')
    #----------------------------------------------------------------------
    def symmetric_difference (self, second_geometry):
        """
        Constructs the geometry that is the union of two geometries minus the
        instersection of those geometries.

        The two input geometries must be the same shape type.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :return: arcgis.gis.Geometry
        """
        res = self._data.symmetric_difference(**{'second_geometry' : second_geometry})
        return pd.Series(res,
                         index=self._index,
                         name='symmetric_difference')
    #----------------------------------------------------------------------
    def touches(self, second_geometry):
        """
        Indicates if the boundaries of the geometries intersect.


        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :return: boolean
        """
        return pd.Series(self._data.touches(**{'second_geometry' : second_geometry}),
                         name='touches',
                         index=self._index)
    #----------------------------------------------------------------------
    def union(self, second_geometry):
        """
        Constructs the geometry that is the set-theoretic union of the input
        geometries.


        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :return: arcgis.gis.Geometry
        """
        res = self._data.union(**{'second_geometry' : second_geometry})
        return pd.Series(res,
                         index=self._index,
                         name='union')
    #----------------------------------------------------------------------
    def within(self, second_geometry, relation=None):
        """
        Indicates if the base geometry is within the comparison geometry.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ---------------     --------------------------------------------------------------------
        relation            Optional String. The spatial relationship type.

                            - BOUNDARY  - Relationship has no restrictions for interiors or boundaries.
                            - CLEMENTINI  - Interiors of geometries must intersect. Specifying CLEMENTINI is equivalent to specifying None. This is the default.
                            - PROPER  - Boundaries of geometries must not intersect.

        ===============     ====================================================================

        :return: boolean

        """
        return pd.Series(self._data.within(**{'second_geometry' : second_geometry,
                                                'relation' : relation}),
                         name='within',
                         index=self._index)


#--------------------------------------------------------------------------
def is_geometry_type(obj):
    t = getattr(obj, 'dtype', obj)
    try:
        return isinstance(t, GeoType) or issubclass(t, GeoType)
    except Exception:
        return False
###########################################################################
@register_dataframe_accessor("spatial")
class GeoAccessor(object):
    """
    The DataFrame Accessor is a namespace that performs dataset operations.
    This includes visualization, spatial indexing, IO and dataset level properties.
    """
    _sr = None
    _viz = None
    _data = None
    _name = None
    _index = None
    _stype = None
    _kdtree = None
    _sindex = None
    _sfname = None
    _renderer = None
    _HASARCPY = None
    _HASSHAPELY = None
    #----------------------------------------------------------------------
    def __init__(self, obj):
        self._data = obj
        self._index = obj.index
        self._name = None
    #----------------------------------------------------------------------
    @property
    def _meta(self):
        """
        Users have the ability to store the source reference back to the
        dataframe.  This will allow the user to compare SeDF with source
        data such as FeatureLayers and Feature Classes.

        ===============   =======================================================
        **Parameter**     **Description**
        ---------------   -------------------------------------------------------
        source            String/Object Reference to the source of the dataframe.
        ===============   =======================================================

        :returns: object/string

        """
        from arcgis.features.geo._tools import _metadata
        if 'metadata' in self._data.attrs and self._data.attrs['metadata'] and \
           isinstance(self._data.attrs['metadata'], _metadata._Metadata):
            return self._data.attrs['metadata']
        else:
            self._meta = _metadata._Metadata()
            return self._meta
    #----------------------------------------------------------------------
    @_meta.setter
    def _meta(self, source):
        """
        Users have the ability to store the source reference back to the
        dataframe.  This will allow the user to compare SeDF with source
        data such as FeatureLayers and Feature Classes.

        ===============   =======================================================
        **Parameter**     **Description**
        ---------------   -------------------------------------------------------
        source            String/Object Reference to the source of the dataframe.
        ===============   =======================================================

        :returns: object/string

        """
        from ._tools import _metadata
        if not 'metadata' in self._data.attrs and \
           isinstance(source, _metadata._Metadata): # creates the attrs entry
            self._data.attrs['metadata'] = source
        elif 'metadata' in self._data.attrs and \
             isinstance(source, _metadata._Metadata) and \
             source != self._data.attrs['metadata']: # sets the new metadata value
            self._data.attrs['metadata'] = source
        elif source is None: # resets/drops the source
            self._data.attrs['metadata'] = _metadata._Metadata()
    #----------------------------------------------------------------------
    @property
    def renderer(self):
        """
        Returns the renderer for the SeDF

        :returns: InsensitiveDict
        """
        if self._meta.renderer is None:
            self._meta.renderer = self._build_renderer()
        return self._meta.renderer
    #----------------------------------------------------------------------
    @renderer.setter
    def renderer(self, renderer):
        """
        Define the renderer for the SeDF.  If none is given, then the value is reset.

        :returns: InsensitiveDict
        """

        if renderer is None:
            renderer = self._build_renderer()
        if isinstance(renderer, dict):
            renderer = InsensitiveDict.from_dict(renderer)
        elif isinstance(renderer, PropertyMap):
            renderer = InsensitiveDict.from_dict(dict(renderer))
        elif isinstance(renderer, InsensitiveDict):
            pass
        else:
            raise ValueError("renderer must be a dictionary type.")
        self._meta.renderer = renderer
    #----------------------------------------------------------------------
    def _build_renderer(self):
        """sets the default symbology"""
        if self._meta.source and \
           hasattr(self._meta.source, 'properties'):
            return self._meta.renderer
        gt = self.geometry_type[0]
        base_renderer = {
            'labelingInfo' : None,
            'label' : "",
            'description' : "",
            'type' : 'simple',
            'symbol' : None
        }
        if gt.lower() in ['point', 'multipoint']:
            base_renderer['symbol'] = {"color":[0,128,0,128],
                         "size":18,"angle":0,
                         "xoffset":0,"yoffset":0,
                         "type":"esriSMS",
                         "style":"esriSMSCircle",
                         "outline":{"color":[0,128,0,255],"width":1,
                                    "type":"esriSLS","style":"esriSLSSolid"}
                         }

        elif gt.lower() =='polyline':
            base_renderer['symbol'] = {
                        "type": "esriSLS",
                        "style": "esriSLSSolid",
                        "color": [0,128,0,128],
                        "width": 1
                    }
        elif gt.lower() =='polygon':
            base_renderer['symbol'] = {
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
        self._meta.renderer = InsensitiveDict(base_renderer)
        return self._meta.renderer
    #----------------------------------------------------------------------
    def _repr_svg_(self):
        """draws the dataframe as SVG features"""

        if self.name:
            fn = lambda g, n: getattr(g, n, None)() if g is not None else None
            vals = np.vectorize(fn, otypes='O')(self._data['SHAPE'], 'svg')
            svg = "\n".join(vals.tolist())
            svg_top = '<svg xmlns="http://www.w3.org/2000/svg" ' \
                'xmlns:xlink="http://www.w3.org/1999/xlink" '
            if len(self._data) == 0:
                return svg_top + '/>'
            else:
                # Establish SVG canvas that will fit all the data + small space
                xmin, ymin, xmax, ymax = self.full_extent
                if xmin == xmax and ymin == ymax:
                    # This is a point; buffer using an arbitrary size
                    xmin, ymin, xmax, ymax = xmin - .001, ymin - .001, xmax + .001, ymax + .001
                else:
                    # Expand bounds by a fraction of the data ranges
                    expand = 0.04  # or 4%, same as R plots
                    widest_part = max([xmax - xmin, ymax - ymin])
                    expand_amount = widest_part * expand
                    xmin -= expand_amount
                    ymin -= expand_amount
                    xmax += expand_amount
                    ymax += expand_amount
                dx = xmax - xmin
                dy = ymax - ymin
                width = min([max([100.0, dx]), 300])
                height = min([max([100.0, dy]), 300])
                try:
                    scale_factor = max([dx, dy]) / max([width, height])
                except ZeroDivisionError:
                    scale_factor = 1
                view_box = "{0} {1} {2} {3}".format(xmin, ymin, dx, dy)
                transform = "matrix(1,0,0,-1,0,{0})".format(ymax + ymin)
                return svg_top + (
                    'width="{1}" height="{2}" viewBox="{0}" '
                    'preserveAspectRatio="xMinYMin meet">'
                    '<g transform="{3}">{4}</g></svg>'
                    ).format(view_box,
                             width,
                             height,
                             transform,
                             svg)
        return
    @staticmethod
    def from_feather(path,
                     spatial_column="SHAPE",
                     columns=None,
                     use_threads=True):
        """
        Load a feather-format object from the file path.

        ======================    =========================================================
        **Argument**              **Description**
        ----------------------    ---------------------------------------------------------
        path                      String. Path object or file-like object. Any valid string
                                  path is acceptable. The string could be a URL. Valid
                                  URL schemes include http, ftp, s3, and file. For file URLs, a host is
                                  expected. A local file could be:

                                  ``file://localhost/path/to/table.feather``.

                                  If you want to pass in a path object, pandas accepts any
                                  ``os.PathLike``.

                                  By file-like object, we refer to objects with a ``read()`` method,
                                  such as a file handler (e.g. via builtin ``open`` function)
                                  or ``StringIO``.
        ----------------------    ---------------------------------------------------------
        spatial_column            Optional String. The default is `SHAPE`. Specifies the column
                                  containing the geo-spatial information.
        ----------------------    ---------------------------------------------------------
        columns                   Sequence/List/Array. The default is `None`.  If not
                                  provided, all columns are read.
        ----------------------    ---------------------------------------------------------
        use_threads               Boolean. The default is `True`. Whether to parallelize
                                  reading using multiple threads.
        ======================    =========================================================

        :returns: pd.DataFrame

        """
        return read_feather(path=path,
                            spatial_column=spatial_column,
                            columns=columns,
                            use_threads=use_threads)
    #----------------------------------------------------------------------
    def set_geometry(self, col, sr=None):
        """Assigns the Geometry Column by Name or by List"""
        from ._array import GeoArray

        if isinstance(col, str) and  \
           col in self._data.columns and \
           self._data[col].dtype.name.lower() != 'geometry':
            idx = self._data[col].first_valid_index()
            if sr is None:
                try:
                    g = self._data.iloc[idx][col]
                    if isinstance(g, dict):
                        self._sr = SpatialReference(Geometry(g['spatialReference']))
                    else:
                        self._sr = SpatialReference(g['spatialReference'])
                except:
                    self._sr = SpatialReference({'wkid' : 4326})
            self._name = col
            q = self._data[col].isna()
            #self._data.loc[q, "SHAPE"] = None
            self._data[col] = GeoArray(self._data[col])
        elif isinstance(col, str) and  \
             col in self._data.columns and \
             self._data[col].dtype.name.lower() == 'geometry':
            self._name = col
            #self._data[col] = self._data[col]
        elif isinstance(col, str) and \
             col not in self._data.columns:
            raise ValueError(
                "Column {name} does not exist".format(name=col))
        elif isinstance(col, pd.Series):
            self._data['SHAPE'] = GeoArray(col.values)
            self._name = "SHAPE"
        elif isinstance(col, GeoArray):
            self._data['SHAPE'] = col
            self._name = "SHAPE"
        elif isinstance(col, (list, tuple)):
            self._data['SHAPE'] = GeoArray(values=col)
            self._name = "SHAPE"
        else:
            raise ValueError(
                "Column {name} is not valid. Please ensure it is of type Geometry".format(name=col))
    #----------------------------------------------------------------------
    @property
    def name(self):
        """returns the name of the geometry column"""
        if self._name is None:
            try:
                cols = [c.lower() for c in self._data.columns.tolist()]
                if any(self._data.dtypes == 'geometry'):
                    name = self._data.dtypes[self._data.dtypes == 'geometry'].index[0]
                    self.set_geometry(name)
                elif "shape" in cols:
                    idx = cols.index("shape")
                    self.set_geometry(self._data.columns[idx])
            except:
                raise Exception("Spatial column not defined, please use `set_geometry`")
        return self._name

    #----------------------------------------------------------------------
    def validate(self, strict=False):
        """
        Determines if the Geo Accessor is Valid with Geometries in all values
        """
        if self._name is None:
            return False
        if strict:
            q = self._data[self.name].notna()
            gt = pd.unique(self._data[q][self.name].geom.geometry_type)
            if len(gt) == 1:
                return True
            else:
                return False
        else:
            q = self._data[self.name].notna()
            return all(pd.unique(self._data[q][self.name].geom.is_valid))
        return True
    #----------------------------------------------------------------------
    def join(self, right_df,
             how='inner', op='intersects',
             left_tag="left", right_tag="right"):
        """
        Joins the current DataFrame to another spatially enabled dataframes based
        on spatial location based.

        .. note::
            requires the SEDF to be in the same coordinate system


        ======================    =========================================================
        **Argument**              **Description**
        ----------------------    ---------------------------------------------------------
        right_df                  Required pd.DataFrame. Spatially enabled dataframe to join.
        ----------------------    ---------------------------------------------------------
        how                       Required string. The type of join:

                                    + `left` - use keys from current dataframe and retains only current geometry column
                                    + `right` - use keys from right_df; retain only right_df geometry column
                                    + `inner` - use intersection of keys from both dfs and retain only current geometry column

        ----------------------    ---------------------------------------------------------
        op                        Required string. The operation to use to perform the join.
                                  The default is `intersects`.

                                  supported perations: `intersects`, `within`, and `contains`
        ----------------------    ---------------------------------------------------------
        left_tag                  Optional String. If the same column is in the left and
                                  right dataframe, this will append that string value to
                                  the field.
        ----------------------    ---------------------------------------------------------
        right_tag                 Optional String. If the same column is in the left and
                                  right dataframe, this will append that string value to
                                  the field.
        ======================    =========================================================

        :returns:
          Spatially enabled Pandas' DataFrame
        """
        allowed_hows = ['left', 'right', 'inner']
        allowed_ops = ['contains', 'within', 'intersects']
        if how not in allowed_hows:
            raise ValueError("`how` is an invalid inputs of %s, but should be %s" % (how, allowed_hows))
        if op not in allowed_ops:
            raise ValueError("`how` is an invalid inputs of %s, but should be %s" % (op, allowed_ops))
        if self.sr != right_df.spatial.sr:
            raise Exception("Difference Spatial References, aborting operation")
        index_left = 'index_{}'.format(left_tag)
        index_right = 'index_{}'.format(right_tag)
        if (any(self._data.columns.isin([index_left, index_right]))
            or any(right_df.columns.isin([index_left, index_right]))):
            raise ValueError("'{0}' and '{1}' cannot be names in the frames being"
                             " joined".format(index_left, index_right))
        # Setup the Indexes in temporary coumns
        #
        left_df = self._data.copy(deep=True)
        left_df.spatial.set_geometry(self.name)
        left_df.reset_index(inplace=True)
        left_df.spatial.set_geometry(self.name)
        # process the right df
        shape_right = right_df.spatial._name
        right_df = right_df.copy(deep=True)
        right_df.reset_index(inplace=True)
        right_df.spatial.set_geometry(shape_right)
        # rename the indexes
        right_df.index = right_df.index.rename(index_right)
        left_df.index = left_df.index.rename(index_left)

        if op == "within":
            # within implemented as the inverse of contains; swap names
            left_df, right_df = right_df, left_df

        tree_idx = right_df.spatial.sindex("quadtree")

        idxmatch = (left_df[self.name]
                    .apply(lambda x: x.extent)
                    .apply(lambda x: list(tree_idx.intersect(x))))
        idxmatch = idxmatch[idxmatch.apply(len) > 0]
        if idxmatch.shape[0] > 0:
            # if output from join has overlapping geometries
            r_idx = np.concatenate(idxmatch.values)
            l_idx = np.concatenate([[i] * len(v) for i, v in idxmatch.iteritems()])

            # Vectorize predicate operations
            def find_intersects(a1, a2):
                return a1.disjoint(a2) == False

            def find_contains(a1, a2):
                return a1.contains(a2)

            predicate_d = {'intersects': find_intersects,
                           'contains': find_contains,
                           'within': find_contains}

            check_predicates = np.vectorize(predicate_d[op])

            result = (
                pd.DataFrame(
                          np.column_stack(
                              [l_idx,
                               r_idx,
                               check_predicates(
                                   left_df[self.name]
                                   .apply(lambda x: x)[l_idx],
                                   right_df[right_df.spatial._name][r_idx])
                               ]))
            )

            result.columns = ['_key_left', '_key_right', 'match_bool']
            result = pd.DataFrame(result[result['match_bool']==1]).drop('match_bool', axis=1)
        else:
            # when output from the join has no overlapping geometries
            result = pd.DataFrame(columns=['_key_left', '_key_right'], dtype=float)
        if op == "within":
            # within implemented as the inverse of contains; swap names
            left_df, right_df = right_df, left_df
            result = result.rename(columns={'_key_left': '_key_right',
                                            '_key_right': '_key_left'})

        if how == 'inner':
            result = result.set_index('_key_left')
            joined = (
                      left_df
                      .merge(result, left_index=True, right_index=True)
                      .merge(right_df.drop(right_df.spatial.name, axis=1),
                          left_on='_key_right', right_index=True,
                          suffixes=('_%s' % left_tag, '_%s' % right_tag))
                     )
            joined = joined.set_index(index_left).drop(['_key_right'], axis=1)
            joined.index.name = None
        elif how == 'left':
            result = result.set_index('_key_left')
            joined = (
                      left_df
                      .merge(result, left_index=True, right_index=True, how='left')
                      .merge(right_df.drop(right_df.spatial.name, axis=1),
                          how='left', left_on='_key_right', right_index=True,
                          suffixes=('_%s' % left_tag, '_%s' % right_tag))
                     )
            joined = joined.set_index(index_left).drop(['_key_right'], axis=1)
            joined.index.name = None
        else:  # 'right join'
            joined = (
                      left_df
                      .drop(left_df.spatial._name, axis=1)
                      .merge(result.merge(right_df,
                          left_on='_key_right', right_index=True,
                          how='right'), left_index=True,
                          right_on='_key_left', how='right')
                      .set_index('index_y')
                     )
            joined = joined.drop(['_key_left', '_key_right'], axis=1)
        try:
            joined.spatial.set_geometry(self.name)
        except:
            raise Exception("Could not create spatially enabled dataframe.")
        joined.reset_index(drop=True, inplace=True)
        return joined
    #----------------------------------------------------------------------
    def plot(self, map_widget=None, **kwargs):
        """

        Plot draws the data on a web map. The user can describe in simple terms how to
        renderer spatial data using symbol.  To make the process simplier a pallette
        for which colors are drawn from can be used instead of explicit colors.


        ======================  =========================================================
        **Explicit Argument**   **Description**
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
        symbol_type             optional string. This is the type of symbol the user
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
        pallette                optional string. The color map to draw from in order to
                                visualize the data.  The default pallette is 'jet'. To
                                get a visual representation of the allowed color maps,
                                use the **display_colormaps** method.
        ----------------------  ---------------------------------------------------------
        alpha                   optional float.  This is a value between 0 and 1 with 1
                                being the default value.  The alpha sets the transparancy
                                of the renderer when applicable.
        ======================  =========================================================

        ** Render Syntax **

        The render syntax allows for users to fully customize symbolizing the data.

        ** Simple Renderer**

        A simple renderer is a renderer that uses one symbol only.

        ======================  =========================================================
        **Optional Argument**   **Description**
        ----------------------  ---------------------------------------------------------
        symbol_type             optional string. This is the type of symbol the user
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
        field1, field2, field3  Attribute field renderer uses to match values.
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



        ** Symbol Syntax **

        =======================  =========================================================
        **Optional Argument**    **Description**
        -----------------------  ---------------------------------------------------------
        symbol_type              optional string. This is the type of symbol the user
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
        from ._viz.mapping import plot

        # small helper to consolidate the plotting function
        def _plot_map_widget(mp_wdgt):
            plot(df=self._data,
                 map_widget=mp_wdgt,
                 name=kwargs.pop('name', "Feature Collection Layer"),
                 renderer_type=kwargs.pop("renderer_type", None),
                 symbol_type=kwargs.pop('symbol_type', None),
                 symbol_style=kwargs.pop('symbol_style', None),
                 col=kwargs.pop('col', None),
                 colors=kwargs.pop('cmap', None) or kwargs.pop('colors', None) or kwargs.pop('pallette', 'jet'),
                 alpha=kwargs.pop('alpha', 1),
                 **kwargs)

        # small helper to address zoom level
        def _adjust_zoom(mp_wdgt):

            # if a single point, the extent will zoom to a scale so large it is almost irrelevant, so back out slightly
            if mp_wdgt.zoom > 16:
                mp_wdgt.zoom = 16

            # if zooming to an extent, it will zoom one level too far, so back out one to make all data visible
            else:
                mp_wdgt.zoom = mp_wdgt.zoom - 1

        # if the map widget is explicitly defined
        if map_widget:
            orig_col = copy.deepcopy(self._data.columns)
            self._data.columns = [c.replace(" ", "_") for c in self._data.columns]
            # plot and be merry
            _plot_map_widget(map_widget)
            self._data.columns = orig_col
            return True

        # otherwise, if a map widget is NOT explicitly defined
        else:

            from arcgis.gis import GIS
            from arcgis.env import active_gis

            # if a gis is not already created in the session, create an anonymous one
            gis = active_gis
            if gis is None:
                gis = GIS()

            # use the GIS to create a map widget
            map_widget = gis.map()

            # plot the data in the map widget
            orig_col = copy.deepcopy(self._data.columns)
            self._data.columns = [c.replace(" ", "_") for c in self._data.columns]
            _plot_map_widget(map_widget)
            self._data.columns = orig_col
            # zoom the map widget to the extent of the data
            map_widget.extent = {
                'spatialReference': self._data.spatial.sr,
                'xmin': self._data.spatial.full_extent[0],
                'ymin': self._data.spatial.full_extent[1],
                'xmax': self._data.spatial.full_extent[2],
                'ymax': self._data.spatial.full_extent[3]
            }

            # adjust the zoom level so the map displays the data as expected
            map_widget.on_draw_end(_adjust_zoom, True)

            # return the map widget so it will be displayed below the cell in Jupyter Notebook
            return map_widget
    #----------------------------------------------------------------------
    def to_featureclass(self, location, overwrite=True, has_z=None, has_m=None, sanitize_columns=True):
        """
        Exports a geo enabled dataframe to a feature class.

        ===========================     ====================================================================
        **Argument**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        location                        Required string. The output of the table.
        ---------------------------     --------------------------------------------------------------------
        overwrite                       Optional Boolean.  If True and if the feature class exists, it will be
                                        deleted and overwritten.  This is default.  If False, the feature class
                                        and the feature class exists, and exception will be raised.
        ---------------------------     --------------------------------------------------------------------
        has_z                           Optional Boolean.  If True, the dataset will be forced to have Z
                                        based geometries.  If a geometry is missing a Z value when true, a
                                        RuntimeError will be raised.  When False, the API will not use the
                                        Z value.
        ---------------------------     --------------------------------------------------------------------
        has_m                           Optional Boolean.  If True, the dataset will be forced to have M
                                        based geometries.  If a geometry is missing a M value when true, a
                                        RuntimeError will be raised. When False, the API will not use the
                                        M value.
        ---------------------------     --------------------------------------------------------------------
        sanitize_columns                Optional Boolean. If True, column names will be converted to string,
                                        invalid characters removed and other checks will be performed. The
                                        default is True.
        ===========================     ====================================================================

        :returns: String

        """
        if location and not str(os.path.dirname(location)).lower() in ['memory', 'in_memory']:
            location = os.path.abspath(path=location)
        return to_featureclass(self,
                               location=location,
                               overwrite=overwrite,
                               has_z=has_z,
                               sanitize_columns=sanitize_columns,
                               has_m=has_m)
    #----------------------------------------------------------------------
    def to_table(self, location, overwrite=True):
        """
        Exports a geo enabled dataframe to a table.

        ===========================     ====================================================================
        **Argument**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        location                        Required string. The output of the table.
        ---------------------------     --------------------------------------------------------------------
        overwrite                       Optional Boolean.  If True and if the table exists, it will be
                                        deleted and overwritten.  This is default.  If False, the table and
                                        the table exists, and exception will be raised.
        ===========================     ====================================================================

        :returns: String

        """
        from arcgis.features.geo._io.fileops import to_table
        from ._tools._utils import run_and_hide
        return run_and_hide(to_table, **{"geo":self,
                                                "location":location,
                                                "overwrite":overwrite})
        #return to_table(geo=self,
        #                location=location,
        #                overwrite=overwrite)

    #----------------------------------------------------------------------
    def to_featurelayer(self,
                        title,
                        gis=None,
                        tags=None,
                        folder=None):
        """
        publishes a spatial dataframe to a new feature layer

        ===========================     ====================================================================
        **Argument**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        title                           Required string. The name of the service
        ---------------------------     --------------------------------------------------------------------
        gis                             Optional GIS. The GIS connection object
        ---------------------------     --------------------------------------------------------------------
        tags                            Optional list of strings. A comma seperated list of descriptive
                                        words for the service.
        ---------------------------     --------------------------------------------------------------------
        folder                          Optional string. Name of the folder where the featurelayer item
                                        and imported data would be stored.
        ===========================     ====================================================================

        :returns: FeatureLayer

        """
        from arcgis import env
        if gis is None:
            gis = env.active_gis
            if gis is None:
                raise ValueError("GIS object must be provided")
        content = gis.content
        return content.import_data(self._data, folder=folder, title=title, tags=tags)
    # ----------------------------------------------------------------------
    @staticmethod
    def from_df(df, address_column="address", geocoder=None, sr=None, geometry_column=None):
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
                                Ignored if the 'geometry_column' is specified.
        --------------------    ---------------------------------------------------------
        geocoder                Optional Geocoder. The geocoder to be used. If not
                                specified, the active GIS's first geocoder is used.
        --------------------    ---------------------------------------------------------
        sr                      Optional integer. The WKID of the spatial reference.
        --------------------    ---------------------------------------------------------
        geometry_column         Optional String.  The name of the geometry column to
                                convert to the arcgis.Geometry Objects (new at version 1.8.1)
        ====================    =========================================================

        :returns: DataFrame



        NOTE: Credits will be consumed for batch_geocoding, from
        the GIS to which the geocoder belongs.

        """
        import arcgis
        from arcgis.geocoding import get_geocoders, geocode, batch_geocode
        from arcgis.geometry import Geometry
        if geometry_column:
            from arcgis.features import GeoAccessor, GeoSeriesAccessor
            if sr is None:
                try:
                    valid_index = df[geometry_column].first_valid_index()
                except:
                    raise ValueError("Column provided is all NULL, please provide a valid column")
                g = Geometry(df[geometry_column].iloc[valid_index])
                sr = g.spatial_reference
                if isinstance(sr, Iterable) and \
                   'wkid' in sr:
                    sr = sr['wkid'] or 4326
                elif isinstance(sr, Iterable) and \
                     'wkt' in sr:
                    sr = sr['wkt'] or 4326
                else:
                    sr = 4326
            from ._array import GeoArray
            df[geometry_column] = GeoArray(df[geometry_column].apply(Geometry))
            df.spatial.set_geometry(geometry_column)
            df.spatial.project(sr)
            return df
        else:

            if geocoder is None:
                geocoder = arcgis.env.active_gis._tools.geocoders[0]
            sr = dict(geocoder.properties.spatialReference)
            geoms = []
            if address_column in df.columns:
                batch_size = geocoder.properties.locatorProperties.MaxBatchSize
                N = len(df)
                geoms = []
                for i in range(0, N, batch_size):
                    start = i
                    stop = i + batch_size if i + batch_size < N else N
                    res = batch_geocode(list(df[start:stop][address_column]), geocoder=geocoder)
                    for index in range(len(res)):
                        try:
                            address = df.loc[start + index, address_column]
                        except: # for older versions, fall back to `df.ix`
                            address = df.ix[start + index, address_column]
                        try:
                            loc = res[index]['location']
                            x = loc['x']
                            y = loc['y']
                            geoms.append(arcgis.geometry.Geometry({'x': x, 'y': y, 'spatialReference': sr}))

                        except:
                            x, y = None, None
                            try:
                                loc = geocode(address, geocoder=geocoder)[0]['location']
                                x = loc['x']
                                y = loc['y']
                            except:
                                print('Unable to geocode address: ' + address)
                                pass
                            geoms.append(None)
            else:
                raise ValueError("Address column not found in dataframe")
            df['SHAPE'] = geoms
            df.spatial.set_geometry("SHAPE")
            return df
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
                                4326 is the default value.
        ====================    =========================================================

        :returns: DataFrame

        """
        from ._io.fileops import _from_xy
        return _from_xy(df=df, x_column=x_column,
                        y_column=y_column, sr=sr)
    #----------------------------------------------------------------------
    @staticmethod
    def from_layer(layer):
        """
        Imports a FeatureLayer to a Spatially Enabled DataFrame

        This operation converts a FeatureLayer or TableLayer to a Pandas' DataFrame

        ====================    =========================================================
        **Argument**            **Description**
        --------------------    ---------------------------------------------------------
        layer                   Required FeatureLayer or TableLayer. The service to convert
                                to a Spatially enabled DataFrame.
        ====================    =========================================================

        Usage:

        >>> from arcgis.features import FeatureLayer
        >>> mylayer = FeatureLayer(("https://sampleserver6.arcgisonline.com/arcgis/rest"
                            "/services/CommercialDamageAssessment/FeatureServer/0"))
        >>> df = from_layer(mylayer)
        >>> print(df.head())

        :returns: Pandas' `DataFrame`

        """
        import json
        try:
            from arcgis.features.geo._io.serviceops import from_layer
            return from_layer(layer=layer)
        except ImportError:
            raise ImportError("Could not load `from_layer`.")
        except json.JSONDecodeError as je:
            raise Exception("Malformed response from server, could not load the dataset: %s" % str(je))
        except Exception as e:
            raise Exception("Could not load the dataset: %s" % str(e))
    #----------------------------------------------------------------------
    @staticmethod
    def from_featureclass(location, **kwargs):
        """
        Returns a Spatially enabled `pandas.DataFrame` from a feature class.

        ===========================     ====================================================================
        **Argument**                    **Description**
        ---------------------------     --------------------------------------------------------------------
        location                        Required string or pathlib.Path. Full path to the feature class
        ===========================     ====================================================================

        *Optional parameters when ArcPy library is available in the current environment*:

        ===========================     ====================================================================
        **Optional Argument**           **Description**
        ---------------------------     --------------------------------------------------------------------
        sql_clause                      sql clause to parse data down. To learn more see
                                        `ArcPy Search Cursor <https://pro.arcgis.com/en/pro-app/arcpy/data-access/searchcursor-class.htm>`_
        ---------------------------     --------------------------------------------------------------------
        where_clause                    where statement. To learn more see `ArcPy SQL reference <https://pro.arcgis.com/en/pro-app/help/mapping/navigation/sql-reference-for-elements-used-in-query-expressions.htm>`_
        ---------------------------     --------------------------------------------------------------------
        fields                          list of strings specifying the field names.
        ---------------------------     --------------------------------------------------------------------
        spatial_filter                  A `Geometry` object that will filter the results.  This requires
                                        `arcpy` to work.
        ===========================     ====================================================================

        :returns: pandas.core.frame.DataFrame
        """
        return from_featureclass(filename=location, **kwargs)
    #----------------------------------------------------------------------
    @staticmethod
    def from_table(filename, **kwargs):
        """
            Allows a user to read from a non-spatial table

            **Note: ArcPy is Required for this method**

            ===============     ====================================================
            **Argument**        **Description**
            ---------------     ----------------------------------------------------
            filename            Required string or pathlib.Path. The path to the
                                table.
            ===============     ====================================================

            **Keyword Arguments**

            ===============     ====================================================
            **Argument**        **Description**
            ---------------     ----------------------------------------------------
            fields              Optional List/Tuple. A list (or tuple) of field
                                names. For a single field, you can use a string
                                instead of a list of strings.

                                Use an asterisk (*) instead of a list of fields if
                                you want to access all fields from the input table
                                (raster and BLOB fields are excluded). However, for
                                faster performance and reliable field order, it is
                                recommended that the list of fields be narrowed to
                                only those that are actually needed.

                                Geometry, raster, and BLOB fields are not supported.

            ---------------     ----------------------------------------------------
            where               Optional String. An optional expression that limits
                                the records returned.
            ---------------     ----------------------------------------------------
            skip_nulls          Optional Boolean. This controls whether records
                                using nulls are skipped.
            ---------------     ----------------------------------------------------
            null_value          Optional String/Integer/Float. Replaces null values
                                from the input with a new value.
            ===============     ====================================================

            :returns: pd.DataFrame
        """
        from arcgis.features.geo._io.fileops import from_table
        return from_table(filename, **kwargs)

    #----------------------------------------------------------------------
    def sindex(self, stype='quadtree', reset=False, **kwargs):
        """
        Creates a spatial index for the given dataset.

        **By default the spatial index is a QuadTree spatial index.**

        If r-tree indexes should be used for large datasets.  This will allow
        users to create very large out of memory indexes.  To use r-tree indexes,
        the r-tree library must be installed.  To do so, install via conda using
        the following command: `conda install -c conda-forge rtree`

        """
        from arcgis.features.geo._index._impl import SpatialIndex
        c = 0
        filename = kwargs.pop('filename', None)
        if reset:
            self._sindex = None
            self._sfname = None
            self._stype = None
        if self._sindex:
            return self._sindex
        #bbox = self.full_extent
        if self.name and \
           filename and \
           os.path.isfile(filename + ".dat") and \
           os.path.isfile(filename + ".idx"):
            l = len(self._data[self.name])
            self._sindex = SpatialIndex(stype=stype,
                                        filename=filename,
                                        bbox=self.full_extent)
            for idx, g in zip(self._index, self._data[self.name]):
                if g:
                    if g.type.lower() == 'point':
                        ge = g.geoextent
                        gext = (ge[0] -.001,ge[1] -.001, ge[2] + .001, ge[3] -.001)
                        self._sindex.insert(oid=idx, bbox=gext)
                    else:
                        self._sindex.insert(oid=idx, bbox=g.geoextent)
                    if c >= int(l/4) + 1:
                        self._sindex.flush()
                        c = 0
                    c += 1
            self._sindex.flush()
            return self._sindex
        elif self.name:
            c = 0
            l = len(self._data[self.name])
            self._sindex = SpatialIndex(stype=stype,
                                        filename=filename,
                                        bbox=self.full_extent)
            for idx, g in zip(self._index, self._data[self.name]):
                if g:
                    if g.type.lower() == 'point':
                        ge = g.geoextent
                        gext = (ge[0] -.001,ge[1] -.001, ge[2] + .001, ge[3] -.001)
                        self._sindex.insert(oid=idx, bbox=gext)
                    else:
                        self._sindex.insert(oid=idx, bbox=g.geoextent)
                    if c >= int(l/4) + 1:
                        self._sindex.flush()
                        c = 0
                    c += 1
            self._sindex.flush()
            return self._sindex
        else:
            raise ValueError(("The Spatial Column must "
                             "be set, call df.spatial.set_geometry."))
    #----------------------------------------------------------------------
    @property
    def __geo_interface__(self):
        """returns the object as an Feature Collection JSON string"""
        template = {
            "type": "FeatureCollection",
            "features": []
        }
        for index, row in self._data.iterrows():
            geom = row[self.name]
            del row[self.name]
            gj = copy.copy(geom.__geo_interface__)
            gj['attributes'] = pd.io.json.loads(pd.io.json.dumps(row)) # ensures the values are converted correctly
            template['features'].append(gj)
        return pd.io.json.dumps(template)
    #----------------------------------------------------------------------
    @property
    def __feature_set__(self):
        """returns a dictionary representation of an Esri FeatureSet"""
        import arcgis
        cols_norm = [col for col in self._data.columns]
        cols_lower = [col.lower() for col in self._data.columns]

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
            "geometryType" : _geom_types[type(self._data[self.name][self._data[self.name].first_valid_index()])],
            "spatialReference" : sr,
            "fields" : [],
            "features" : []
        }
        # Ensure all number values are 0 so errors do not occur.
        for c in self._data.select_dtypes(include='number').columns.tolist():
            self._data[c].fillna(0, inplace=True)

        if 'objectid' in cols_lower:
            fs['objectIdFieldName'] = cols_norm[cols_lower.index('objectid')]
            fs['displayFieldName'] = cols_norm[cols_lower.index('objectid')]
            if self._data[fs['objectIdFieldName']].is_unique == False:
                old_series = self._data[fs['objectIdFieldName']].copy()
                self._data[fs['objectIdFieldName']] = list(range(1, self._data.shape[0] + 1))
                res = self.__feature_set__
                self._data[fs['objectIdFieldName']] = old_series
                return res
        elif 'fid' in cols_lower:
            fs['objectIdFieldName'] = cols_norm[cols_lower.index('fid')]
            fs['displayFieldName'] = cols_norm[cols_lower.index('fid')]
            if self._data[fs['objectIdFieldName']].is_unique == False:
                old_series = self._data[fs['objectIdFieldName']].copy()
                self._data[fs['objectIdFieldName']] = list(range(1, self._data.shape[0] + 1))
                res = self.__feature_set__
                self._data[fs['objectIdFieldName']] = old_series
                return res
        elif 'oid' in cols_lower:
            fs['objectIdFieldName'] = cols_norm[cols_lower.index('oid')]
            fs['displayFieldName'] = cols_norm[cols_lower.index('oid')]
            if self._data[fs['objectIdFieldName']].is_unique == False:
                old_series = self._data[fs['objectIdFieldName']].copy()
                self._data[fs['objectIdFieldName']] = list(range(1, self._data.shape[0] + 1))
                res = self.__feature_set__
                self._data[fs['objectIdFieldName']] = old_series
                return res
        else:
            self._data['OBJECTID'] = list(range(1, self._data.shape[0] + 1))
            res = self.__feature_set__
            del self._data['OBJECTID']
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
        if self.name in cols_norm:
            cols_norm.pop(cols_norm.index(self.name))
        for col in cols_norm:
            try:
                idx = self._data[col].first_valid_index()
                col_val = self._data[col].loc[idx]
            except:
                col_val = ""
            if isinstance(col_val, (str, np.str)):
                l = self._data[col].str.len().max()
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
                                      )):#pd.datetime
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
        for row in self._data.to_dict('records'):
            geom = {}
            if self.name in row:
                geom = row[self.name]
                del row[self.name]
            for f in date_fields:
                try:
                    row[f] = int(row[f].to_pydatetime().timestamp() * 1000)
                except:
                    row[f] = None
            if geom and pd.notna(geom):


                features.append(
                    {
                        "geometry" : dict(geom),
                        "attributes" : row
                    })
            elif pd.notna(geom) == False:
                features.append(
                    {
                        "geometry" : None,
                        "attributes" : row
                    })
            else:
                features.append(
                    {
                        "geometry" : geom,
                        "attributes" : row
                    })
            del row
            del geom
        fs['features'] = features
        return fs
    #----------------------------------------------------------------------
    def _check_geometry_engine(self):
        if self._HASARCPY is None:
            try:
                import arcpy
                self._HASARCPY = True
            except:
                self._HASARCPY = False
        if self._HASSHAPELY is None:
            try:
                import shapely
                self._HASSHAPELY = True
            except:
                self._HASSHAPELY = False
        return self._HASARCPY, self._HASSHAPELY
    #----------------------------------------------------------------------
    @property
    def sr(self):
        """gets/sets the spatial reference of the dataframe"""
        data = [getattr(g, 'spatialReference', None) or g['spatialReference'] \
                for g in self._data[self.name] \
                if g not in [None, np.NaN, np.nan, ''] and isinstance(g, dict)]
        srs = [SpatialReference(sr) for sr in pd.DataFrame(data).drop_duplicates().to_dict('records')]
        if len(srs) == 1:
            return srs[0]
        return srs
    #----------------------------------------------------------------------
    @sr.setter
    def sr(self, ref):
        """Sets the spatial reference"""
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY:
            try:
                sr = self.sr
            except:
                sr = None
            if sr and \
               'wkid' in sr:
                wkid = sr['wkid']
            if sr and \
               'wkt' in sr:
                wkt = sr['wkt']
            if isinstance(ref, (dict, SpatialReference)) and \
               sr is None:
                self._data[self.name] = self._data[self.name].geom.project_as(ref)
            elif isinstance(ref, SpatialReference):
                if ref != sr:
                    self._data[self.name] = self._data[self.name].geom.project_as(ref)
            elif isinstance(ref, int):
                if ref != wkid:
                    self._data[self.name] = self._data[self.name].geom.project_as(ref)
            elif isinstance(ref, str):
                if ref != wkt:
                    self._data[self.name] = self._data[self.name].geom.project_as(ref)
            elif isinstance(ref, dict):
                nsr = SpatialReference(ref)
                if sr != nsr:
                    self._data[self.name] = self._data[self.name].geom.project_as(ref)
        else:
            if ref:
                if isinstance(ref, str):
                    ref = {"wkt" : ref}
                elif isinstance(ref, int):
                    ref = {"wkid" : ref}
                self._data[self.name].apply(lambda x: x.update({'spatialReference': ref}) if pd.notnull(x) else None)
    #----------------------------------------------------------------------
    def to_featureset(self):
        """
        Converts a spatial dataframe to a feature set object
        """
        from arcgis.features import FeatureSet
        return FeatureSet.from_dataframe(self._data)
    #----------------------------------------------------------------------
    def to_feature_collection(self,
                              name=None,
                              drawing_info=None,
                              extent=None,
                              global_id_field=None):
        """
        Converts a spatially enabled pd.DataFrame to a Feature Collection

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
        import string
        import random

        if name is None:
            name = random.choice(string.ascii_letters) + uuid.uuid4().hex[:5]
        template = {
            'showLegend' : True,
            'layers' : []
        }
        if extent is None:
            ext = self.full_extent
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
            import json
            di = { 'renderer' : json.loads(self._data.spatial.renderer.json) }
        else:
            di = drawing_info
        layer = {'layerDefinition': {'currentVersion': 10.7,
                                     'id': 0,
                                     'name': name,
                                     'type': 'Feature Layer',
                                     'displayField': '',
                                     'description': '',
                                     'copyrightText': '',
                                     'defaultVisibility': True,
                                     'relationships': [],
                                     'isDataVersioned': False,
                                     'supportsAppend': True,
                                     'supportsCalculate': True,
                                     'supportsASyncCalculate': True,
                                     'supportsTruncate': False,
                                     'supportsAttachmentsByUploadId': True,
                                     'supportsAttachmentsResizing': True,
                                     'supportsRollbackOnFailureParameter': True,
                                     'supportsStatistics': True,
                                     'supportsExceedsLimitStatistics': True,
                                     'supportsAdvancedQueries': True,
                                     'supportsValidateSql': True,
                                     'supportsCoordinatesQuantization': True,
                                     'supportsFieldDescriptionProperty': True,
                                     'supportsQuantizationEditMode': True,
                                     'supportsApplyEditsWithGlobalIds': False,
                                     'supportsMultiScaleGeometry': True,
                                     'supportsReturningQueryGeometry': True,
                                     'hasGeometryProperties': True,
                                     'advancedQueryCapabilities': {
                                         'supportsPagination': True,
                                         'supportsPaginationOnAggregatedQueries': True,
                                         'supportsQueryRelatedPagination': True,
                                         'supportsQueryWithDistance': True,
                                         'supportsReturningQueryExtent': True,
                                         'supportsStatistics': True,
                                         'supportsOrderBy': True,
                                         'supportsDistinct': True,
                                         'supportsQueryWithResultType': True,
                                         'supportsSqlExpression': True,
                                         'supportsAdvancedQueryRelated': True,
                                         'supportsCountDistinct': True,
                                         'supportsReturningGeometryCentroid': True,
                                         'supportsReturningGeometryProperties': True,
                                         'supportsQueryWithDatumTransformation': True,
                                         'supportsHavingClause': True,
                                         'supportsOutFieldSQLExpression': True,
                                         'supportsMaxRecordCountFactor': True,
                                         'supportsTopFeaturesQuery': True,
                                         'supportsDisjointSpatialRel': True,
                                         'supportsQueryWithCacheHint': True},
                                     'useStandardizedQueries': False,
                                     'geometryType': fs['geometryType'],
                                     'minScale': 0,
                                     'maxScale': 0,
                                     'extent': extent,
                                     'drawingInfo': di,
                                     'allowGeometryUpdates': True,
                                     'hasAttachments': False,
                                     'htmlPopupType': 'esriServerHTMLPopupTypeNone',
                                     'hasM': False,
                                     'hasZ': False,
                                     'objectIdField': fs['objectIdFieldName'] or "OBJECTID",
                                     'globalIdField': '',
                                     'typeIdField': '',
                                     'fields': fs['fields'],
                                     'types': [],
                                     'supportedQueryFormats': 'JSON, geoJSON',
                                     'hasStaticData': True,
                                     'maxRecordCount': 32000,
                                     'standardMaxRecordCount': 4000,
                                     'tileMaxRecordCount': 4000,
                                     'maxRecordCountFactor': 1,
                                     'capabilities': 'Query'},
                 'featureSet':  {
                     'features' : fs['features'],
                     'geometryType' : fs['geometryType']
                 }
                 }
        if global_id_field is not None:
            layer['layerDefinition']['globalIdField'] = global_id_field
        return FeatureCollection(layer)

    # ---------------------------------------------------------------------

    @staticmethod
    def from_geodataframe(geo_df, inplace=False, column_name="SHAPE"):
        """
        Import Geopandas GeoDataFrame into an ArcGIS Spatially enabled DataFrame.
        Requires geopandas library be installed in current environment.

        =====================  ===============================================================
        **Argument**           **Description**
        ---------------------  ---------------------------------------------------------------
        geo_df                 GeoDataFrame object, created using GeoPandas library
        ---------------------  ---------------------------------------------------------------
        inplace                Optional Bool. When True, the existing GeoDataFrame is spatially
                                enabled and returned. When False, a new Spatially Enabled
                                DataFrame object is returned. Default is False.
        ---------------------  ---------------------------------------------------------------
        column_name            Optional String. Sets the name of the geometry column. Default
                                is `SHAPE`.
        =====================  ===============================================================

        :return: ArcGIS Spatially Enabled DataFrame object.
        """
        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError('Requires Geopandas library installed for this functionality')

        # import geometry libraries
        from arcgis.geometry import Geometry as ags_geometry
        from arcgis.features.geo._array import GeoArray

        # import pandas
        import pandas as pd
        import numpy as np

        # get wkid
        try:
            if geo_df.crs is not None and hasattr(geo_df.crs, 'to_epsg'):
                # check for pyproj
                epsg_code = geo_df.crs.to_epsg()
            elif geo_df.crs is not None and 'init' in geo_df.crs:
                epsg_code = geo_df.crs['init'].split(':')[-1]
                epsg_code = int(epsg_code) # convert string to number
            elif geo_df.crs is not None:
                # crs is present, but no epsg code. Try to reproject to 4326
                geo_df.to_crs(epsg=4326, inplace=True)
                epsg_code = 4326
            else:
                _LOGGER.info('Cannot acquire spatial reference from GeoDataFrame. Setting it a default of WKID 4326')
                epsg_code = 4326 # set a safe default value

        except Exception as proj_ex:
            _LOGGER.warning('Error acquiring spatial reference from GeoDataFrame' \
                            ' Spatial reference will not be set.' + str(proj_ex))
            epsg_code = None

        if epsg_code:
            spatial_reference = {'wkid':epsg_code}
        else:
            spatial_reference = None

        # convert geometry
        def _converter(g):
            if g is not None:
                # return ags_geometry(shp_mapping(g))
                return ags_geometry.from_shapely(g, spatial_reference=spatial_reference)
            else:
                return None

        # vectorize converter so it will run efficiently on GeoSeries - avoids loops
        v_func = np.vectorize(_converter, otypes='O')

        # initialize empty array
        ags_geom = np.empty(geo_df.shape[0], dtype="O")

        ags_geom[:] = v_func(geo_df[geo_df.geometry.name].values)

        if inplace:
            geo_df[column_name] = GeoArray(ags_geom)
        else:
            geo_df = pd.DataFrame(geo_df.drop(columns=geo_df.geometry.name))
            geo_df[column_name] = GeoArray(ags_geom)

        geo_df.spatial.set_geometry(column_name)
        geo_df.spatial.sr = spatial_reference

        return geo_df

    # ---------------------------------------------------------------------

    @property
    def full_extent(self):
        """
        Returns the extent of the dataframe

        :returns: tuple

        >>> df.spatial.full_extent
        (-118, 32, -97, 33)

        """
        ge = self._data[self.name].geom.extent
        q = ge.notnull()
        data = ge[q].tolist()
        array = np.array(data)
        return (float(array[:,0][array[:,0]!=None].min()),
                float(array[:,1][array[:,1]!=None].min()),
                float(array[:,2][array[:,2]!=None].max()),
                float(array[:,3][array[:,3]!=None].max()))
    #----------------------------------------------------------------------
    @property
    def area(self):
        """
        Returns the total area of the dataframe

        :returns: float

        >>> df.spatial.area
        143.23427

        """
        return self._data[self.name].values.area.sum()
    #----------------------------------------------------------------------
    @property
    def length(self):
        """
        Returns the total length of the dataframe

        :returns: float

        >>> df.spatial.length
        1.23427

        """
        return self._data[self.name].values.length.sum()
    #----------------------------------------------------------------------
    @property
    def centroid(self):
        """
        Returns the centroid of the dataframe

        :returns: Geometry

        >>> df.spatial.centroid
        (-14.23427, 39)

        """
        q = self._data[self.name].geom.centroid.isnull()
        df = pd.DataFrame(self._data[~q][self.name].geom.centroid.tolist(), columns=['x','y'])
        return df['x'].mean(), df['y'].mean()
    #----------------------------------------------------------------------
    @property
    def true_centroid(self):
        """
        Returns the true centroid of the dataframe

        :returns: Geometry

        >>> df.spatial.true_centroid
        (1.23427, 34)

        """
        q = self._data[self.name].notnull()
        df = pd.DataFrame(data=self._data[self.name][q].geom.true_centroid.tolist(), columns=['x','y']).mean()
        return df['x'], df['y']
    #----------------------------------------------------------------------
    @property
    def geometry_type(self):
        """
        Returns a list Geometry Types for the DataFrame
        """
        gt = self._data[self.name].geom.geometry_type
        return pd.unique(gt).tolist()
    #----------------------------------------------------------------------
    @property
    def has_z(self):
        """
        Returns a boolean that determines if the datasets have `Z` values

        :returns: Boolean
        """
        return self._data[self.name].geom.has_z.all()
    #----------------------------------------------------------------------
    @property
    def has_m(self):
        """
        Returns a boolean that determines if the datasets have `Z` values

        :returns: Boolean
        """
        return self._data[self.name].geom.has_m.all()
    #----------------------------------------------------------------------
    @property
    def bbox(self):
        """
        Returns the total length of the dataframe

        :returns: Polygon

        >>> df.spatial.bbox
        {'rings' : [[[1,2], [2,3], [3,3],....]], 'spatialReference' {'wkid': 4326}}
        """
        xmin, ymin, xmax, ymax = self.full_extent
        sr = self.sr
        if isinstance(sr, list) and \
           len(sr) > 0:
            sr = sr[0]
        if xmin == xmax:
            xmin -= .001
            xmax += .001
        if ymin == ymax:
            ymin -= .001
            ymax += .001
        return Geometry(
            {'rings' : [[[xmin,ymin], [xmin, ymax],
                         [xmax, ymax], [xmax, ymin],
                         [xmin, ymin]]],
             'spatialReference' : dict(sr)})
    #----------------------------------------------------------------------
    def distance_matrix(self, leaf_size=16, rebuild=False):
        """
        Creates a k-d tree to calculate the nearest-neighbor problem.

        **requires scipy**

        ====================     ====================================================================
        **Argument**             **Description**
        --------------------     --------------------------------------------------------------------
        leafsize                 Optional Integer. The number of points at which the algorithm
                                 switches over to brute-force. Default: 16.
        --------------------     --------------------------------------------------------------------
        rebuild                  Optional Boolean. If True, the current KDTree is erased. If false,
                                 any KD-Tree that exists will be returned.
        ====================     ====================================================================


        :returns: scipy's KDTree class

        """
        _HASARCPY, _HASSHAPELY = self._check_geometry_engine()
        if _HASARCPY == False and _HASSHAPELY == False:
            return None
        if rebuild:
            self._kdtree = None
        if self._kdtree is None:
            try:
                from scipy.spatial import cKDTree as KDTree
            except ImportError:
                from scipy.spatial import KDTree
            xy = self._data[self.name].geom.centroid.tolist()
            self._kdtree = KDTree(data=xy, leafsize=leaf_size)
            return self._kdtree
        else:
            return self._kdtree
    #----------------------------------------------------------------------
    def select(self, other):
        """
        This operation performs a dataset wide **selection** by geometric
        intersection. A geometry or another Spatially enabled DataFrame
        can be given and `select` will return all rows that intersect that
        input geometry.  The `select` operation uses a spatial index to
        complete the task, so if it is not built before the first run, the
        function will build a quadtree index on the fly.

        **requires ArcPy or Shapely**

        :returns: pd.DataFrame (spatially enabled)

        """
        from arcgis.features.geo._tools import select
        return select(sdf=self._data, other=other)
    #----------------------------------------------------------------------
    def overlay(self, sdf, op="union"):
        """
        Performs spatial operation operations on two spatially enabled dataframes.

        **requires ArcPy or Shapely**

        =========================    =========================================================
        **Argument**                 **Description**
        -------------------------    ---------------------------------------------------------
        sdf                          Required Spatially Enabled DataFrame. The geometry to
                                     perform the operation from.
        -------------------------    ---------------------------------------------------------
        op                           Optional String. The spatial operation to perform.  The
                                     allowed value are: union, erase, identity, intersection.
                                     `union` is the default operation.
        =========================    =========================================================

        :returns: Spatially enabled DataFrame (pd.DataFrame)

        """
        from arcgis.features.geo._tools import overlay
        return overlay(sdf1=self._data, sdf2=sdf, op=op.lower())
    #----------------------------------------------------------------------
    def relationship(self, other, op, relation=None):
        """
        This method allows for dataframe to dataframe compairson using
        spatial relationships.  The return is a pd.DataFrame that meet the
        operations' requirements.

        =========================    =========================================================
        **Argument**                 **Description**
        -------------------------    ---------------------------------------------------------
        other                        Required Spatially Enabled DataFrame. The geometry to
                                     perform the operation from.

        -------------------------    ---------------------------------------------------------
        op                           Optional String. The spatial operation to perform.  The
                                     allowed value are: contains,crosses,disjoint,equals,
                                     overlaps,touches, or within.

                                     - contains - Indicates if the base geometry contains the comparison geometry.
                                     - crosses -  Indicates if the two geometries intersect in a geometry of a lesser shape type.
                                     - disjoint - Indicates if the base and comparison geometries share no points in common.
                                     - equals - Indicates if the base and comparison geometries are of the same shape type and define the same set of points in the plane. This is a 2D comparison only; M and Z values are ignored.
                                     - overlaps - Indicates if the intersection of the two geometries has the same shape type as one of the input geometries and is not equivalent to either of the input geometries.
                                     - touches - Indicates if the boundaries of the geometries intersect.
                                     - within - Indicates if the base geometry is within the comparison geometry.
                                     - intersect - Intdicates if the base geometry has an intersection of the other geometry.
        -------------------------    ---------------------------------------------------------
        relation                     Optional String.  The spatial relationship type.  The
                                     allowed values are: BOUNDARY, CLEMENTINI, and PROPER.

                                     + BOUNDARY - Relationship has no restrictions for interiors or boundaries.
                                     + CLEMENTINI - Interiors of geometries must intersect. This is the default.
                                     + PROPER - Boundaries of geometries must not intersect.

                                     This only applies to contains,
        =========================    =========================================================

        :returns: Spatially enabled DataFrame (pd.DataFrame)


        """
        from ._tools import contains, crosses, disjoint
        from ._tools import equals, overlaps, touches
        from ._tools import within
        _ops_allowed = {'contains' : contains,
                        'crosses': crosses,
                        'disjoint': disjoint,
                        'intersect': disjoint,
                        'equals': equals,
                        'overlaps' : overlaps,
                        'touches': touches,
                        'within' : contains}

        if not op.lower() in _ops_allowed.keys():
            raise ValueError("Invalid `op`. Please use a proper operation.")

        if op.lower() in ['contains', 'within']:
            fn = _ops_allowed[op.lower()]
            return fn(sdf=self._data, other=other, relation=relation)
        elif op.lower() in ['intersect']:
            fn = _ops_allowed[op.lower()]
            return fn(sdf=self._data, other=other) == False
        else:
            fn = _ops_allowed[op.lower()]
            return fn(sdf=self._data, other=other)
    #----------------------------------------------------------------------
    def voronoi(self):
        """
        Generates a voronoi diagram on the whole dataset.  If the geometry
        is not a `Point` then the centroid is used for the geometry.  The
        result is a polygon `GeoArray` Series that matches 1:1 to the original
        dataset.

        **requires scipy**

        :returns: pd.Series

        """
        _HASARCPY, _HASSHAPELY = self._check_geometry_engine()
        if _HASARCPY == False and _HASSHAPELY == False:
            return None
        radius = max(abs(self.full_extent[0] - self.full_extent[2]),
                     abs(self.full_extent[1] - self.full_extent[3]))
        from ._array import GeoArray
        from scipy.spatial import Voronoi
        xy = self._data[self.name].geom.centroid
        vor = Voronoi(xy.tolist())
        if vor.points.shape[1] != 2:
            raise ValueError("Supports 2-D only.")
        new_regions = []
        new_vertices = vor.vertices.tolist()
        center = vor.points.mean(axis=0)
        # Construct a map containing all ridges for a
        # given point
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points,
                                      vor.ridge_vertices):
            all_ridges.setdefault(
                p1, []).append((p2, v1, v2))
            all_ridges.setdefault(
                p2, []).append((p1, v1, v2))
        # Reconstruct infinite regions
        for p1, region in enumerate(vor.point_region):
            vertices = vor.regions[region]
            if all(v >= 0 for v in vertices):
                # finite region
                new_regions.append(vertices)
                continue
            # reconstruct a non-finite region
            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]
            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    # finite ridge: already in the region
                    continue
                # Compute the missing endpoint of an
                # infinite ridge
                t = vor.points[p2] - \
                    vor.points[p1]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal
                midpoint = vor.points[[p1, p2]]. \
                    mean(axis=0)
                direction = np.sign(
                    np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v2] + \
                    direction * radius
                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())
            # Sort region counterclockwise.
            vs = np.asarray([new_vertices[v]
                             for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(
                vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region = np.array(new_region)[
                np.argsort(angles)]
            new_regions.append(new_region.tolist())
        sr = self.sr
        return pd.Series(GeoArray([Geometry({'rings' : [[new_vertices[l] for l in r]],
                                             'spatialReference' : sr}).buffer(0) \
                                   for r in new_regions]))
    #----------------------------------------------------------------------
    def project(self, spatial_reference, transformation_name=None):
        """
        Reprojects the who dataset into a new spatial reference. This is an inplace operation meaning
        that it will update the defined geometry column from the `set_geometry`.

        **This requires ArcPy or pyproj v4**

        ====================     ====================================================================
        **Argument**             **Description**
        --------------------     --------------------------------------------------------------------
        spatial_reference        Required SpatialReference. The new spatial reference. This can be a
                                 SpatialReference object or the coordinate system name.
        --------------------     --------------------------------------------------------------------
        transformation_name      Optional String. The geotransformation name.
        ====================     ====================================================================

        :returns: boolean
        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        HASPYPROJ = True
        try:
            import imp
            imp.find_module('pyproj')
        except ImportError:
            HASPYPROJ = False
        try:

            if isinstance(spatial_reference, (int, str)) and HASARCPY:
                import arcpy
                spatial_reference = arcpy.SpatialReference(spatial_reference)
                vals = self._data[self.name].values.project_as(**{'spatial_reference' : spatial_reference,
                                                                  'transformation_name' : transformation_name})
                self._data[self.name] = vals
                return True
            elif isinstance(spatial_reference, (int, str)) and HASPYPROJ:
                vals = self._data[self.name].values.project_as(**{'spatial_reference' : spatial_reference,
                                                                  'transformation_name' : transformation_name})
                self._data[self.name] = vals
                return True
            else:
                return False
        except Exception as e:
            raise Exception(e)

    def sanitize_column_names(self, convert_to_string=True, remove_special_char=True, inplace=False,
                              use_snake_case=True):
        """
        Cleans column names by converting them to string, removing special characters, renaming columns without
        column names to 'noname', renaming duplicates with integer suffixes and switching spaces or Pascal or
        camel cases to Python's favored snake_case style.

        Snake_casing gives you consistent column names, no matter what the flavor of your backend database is
        when you publish the DataFrame as a Feature Layer in your web GIS.

        ==============================     ====================================================================
        **Argument**                       **Description**
        ------------------------------     --------------------------------------------------------------------
        convert_to_string                  Optional Boolean. Default is True. Converts column names to string
        ------------------------------     --------------------------------------------------------------------
        remove_special_char                Optional Boolean. Default is True. Removes any characters in column
                                           names that are not numeric or underscores. This also ensures column
                                           names begin with alphabets by removing numeral prefixes.
        ------------------------------     --------------------------------------------------------------------
        inplace                            Optional Boolean. Default is False. If True, edits the DataFrame
                                           in place and returns Nothing. If False, returns a new DataFrame object.
        ------------------------------     --------------------------------------------------------------------
        use_snake_case                     Optional Boolean. Default is True. Makes column names lower case,
                                           and replaces spaces between words with underscores. If column names
                                           are in PascalCase or camelCase, it replaces them to snake_case.
        ==============================     ====================================================================

        :returns: pd.DataFrame object if inplace=False. Else None.
        """

        return _sanitize_column_names(self, convert_to_string, remove_special_char, inplace, use_snake_case)
