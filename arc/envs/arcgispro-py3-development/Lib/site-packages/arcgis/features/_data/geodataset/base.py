"""
This module contains the base functions and properties all
spatial dataframes will inherit from.

"""
from __future__ import print_function
from __future__ import division
from six import integer_types
import numpy as np
import pandas as pd
from arcgis.geometry import _types
from pandas import DataFrame, Series
GEOTYPES = [_types.Geometry]
try:
    import arcpy
    HASARCPY  = True
    GEOTYPES.append(arcpy.Geometry)
except:
    GEOTYPES = [_types.Geometry]
    HASARCPY = False

try:
    import shapely
    from shapely.geometry.base import BaseGeometry as _BaseGeometry
    HASSHAPELY = True
    GEOTYPES.append(_BaseGeometry)
except ImportError:
    HASSHAPELY = False

from warnings import warn
try:
    from .index.quadtree import Index as QuadIndex
    HAS_QUADINDEX = True
except ImportError:
    HAS_QUADINDEX = False
try:
    from .index.rtree import RTreeError, Rect
    from .index.si import SpatialIndex
    HAS_SINDEX = True
except ImportError:
    class RTreeError(Exception):
        pass
    HAS_SINDEX = False

GEOTYPES = tuple(GEOTYPES)

#--------------------------------------------------------------------------
def _call_property(this, op, null_value=None, isGeoseries=False):
    """
    calls a property by name on an object
    Inputs:
     this: object to call the function on
     op: string, name of the function to call
     null_value: if the function requires a specific return value on error
      or failure, this value will be returned.
     geoseries: if true, a spatial series is returned else a regular pandas
      series object
    Output:
     Returns an value from the function or null value.
    raises: ValueError
    """
    if isGeoseries:
        from . import GeoSeries
        a = [getattr(geom, op, null_value) for geom in this.geometry \
                 if hasattr(geom, op)]#hasattr(geom, 'as_arcpy') and \
        return GeoSeries(a,
                      index=this.index)
        #[getattr(geom, op, null_value) for geom in this.geometry if hasattr(geom.as_arcpy, op)],
    else:
        a = [getattr(geom, op, null_value) for geom in this.geometry \
             if hasattr(geom, op)]
        return Series(a,
                      index=this.index)
        #[getattr(geom.as_arcpy, op, null_value) for geom in this.geometry if hasattr(geom.as_arcpy, op)],

    return null_value
#--------------------------------------------------------------------------
def _call_function(this, op, second_geometry=None,
                   null_value=None, isGeoseries=False,
                   **kwargs):
    """
    Calls a function on a given object.
    Inputs:
     this: object to call the function on
     op: string, name of the function to call
     second_geometry: arcpy.Geometry object
     null_value: if the function requires a specific return value on error
      or failure, this value will be returned.
     geoseries: if true, a spatial series is returned else a regular pandas
      series object
     kwargs: optional values to be passed to the function
    Output:
     Returns an value from the function or null value.
    raises: ValueError

    """
    from .geoseries import GeoSeries
    other = None
    if 'other' in kwargs and \
       second_geometry is None:
        other = kwargs.pop('other')
        hasOther = True
        hasSecondGeom = False
        isGSeries = isinstance(other, BaseSpatialPandas)
        isPSeries = isinstance(other, Series)
        isGeom = isinstance(other, GEOTYPES)
    elif second_geometry is not None and \
         'other' not in kwargs:
        hasOther = False
        hasSecondGeom = True
        isPSeries = isinstance(second_geometry, Series)
        isSeries = isinstance(second_geometry, BaseSpatialPandas)
        isGeom = isinstance(second_geometry, GEOTYPES)
    elif 'other' in kwargs and second_geometry is not None:
        raise ValueError("Two geometries given as other and second_geometry, you can only have one")
    else: # default
        other = None
        hasOther = False
        hasSecondGeom = False
        isGeom = False
        isGSeries = False
        isPSeries = False

    if isGeoseries:
        if isPSeries or \
           (other and isinstance(other, BaseSpatialPandas)):
            this = this.geometry
            if second_geometry is not None:
                other = second_geometry
                key = 'second_geometry'
            elif other is not None:
                key = 'other'
            else:
                key = 'no_geom'
            sr = this.sr
            this, other = this.align(other.geometry)
            vals = []
            for geom, other_geom in zip(this, other):
                fn = getattr(geom, op)
                if key == 'other':
                    kwargs['other'] = other_geom
                    vals.append(fn(**kwargs))
                elif key == 'second_geometry':
                    vals.append(fn(second_geometry=second_geometry, **kwargs))
                else:
                    vals.append(fn(**kwargs))
                del fn
                del geom
                del other_geom
            return GeoSeries(np.array(vals), index=this.index)
        elif isGeom:
            if second_geometry:
                return GeoSeries([getattr(s, op)(second_geometry=second_geometry,
                                                 **kwargs)
                                  for s in this.geometry],
                                 index=this.index, )
            else:
                if hasOther:
                    kwargs['other'] = other
                vals = [getattr(s, op)(**kwargs) for s in this.geometry]
                return GeoSeries([getattr(s, op)(**kwargs)
                                  for s in this.geometry],
                                 index=this.index, )
        else:
            return GeoSeries([getattr(s, op)(**kwargs)
                              for s in this.geometry],
                             index=this.index, )
    else:
        if isPSeries:
            this = this.geometry
            if second_geometry is not None:
                other = second_geometry
                key = 'second_geometry'
            elif other is not None:
                key = 'other'
            else:
                key = 'no_geom'
            sr = this.sr
            this, other = this.align(other.geometry)
            vals = []
            for geom, other_geom in zip(this, other):
                fn = getattr(geom, op)
                if key == 'other':
                    kwargs['other'] = other_geom
                    vals.append(fn(**kwargs))
                elif key == 'second_geometry':
                    vals.append(fn(second_geometry=other_geom, **kwargs))
                else:
                    vals.append(fn(**kwargs))
                del fn
                del geom
                del other_geom
            return Series(np.array(vals), index=this.index)
        fn = None
        vals = []
        for s in this.geometry:
            if hasattr(s, op):
                fn = getattr(s, op)
                if second_geometry:
                    vals.append(fn(second_geometry=second_geometry,**kwargs))
                else:
                    vals.append(fn(**kwargs))
            elif s and hasattr(s, op)  == False:
                raise ValueError("Invalid operation: %s" % op)
            else:
                vals.append(np.nan)
            del s
        return Series(vals, index=this.index)
    return null_value
########################################################################
class BaseSpatialPandas(object):
    """
    Base object that the Series and DataFrame will inherit from.
    """

    #----------------------------------------------------------------------
    def _generate_sindex(self):
        self._sindex = None
        if not HAS_SINDEX and not HAS_QUADINDEX:
            warn("Cannot generate spatial index: Missing package 'rtree'.")
        elif HAS_QUADINDEX:
            bbox = self.series_extent
            if bbox is None:
                bbox = [-180, -90, 180, 90]
            qi = QuadIndex(bbox=bbox)
            geometry_type = self.geometry_type.lower()
            if geometry_type == 'point':
                geometry_type = self.geometry[self.geometry.index[0]].type.lower()
            for i, (idx, item) in enumerate(self.geometry.iteritems()):
                if pd.notnull(item) and item:
                    if geometry_type in ('point', 'pointgeometry'):
                        factor = .01
                    else:
                        factor = 0
                    if geometry_type == 'pointgeometry':
                        item = item.centroid
                    if HASARCPY:
                        try:
                            xmin, ymin, xmax, ymax = item.extent
                            qi.insert(item=idx, bbox=(xmin - factor,
                                                      ymin - factor,
                                                      xmax + factor,
                                                      ymax + factor))
                        except:
                            pass
                    else:
                        try:
                            qi.insert(item=idx, bbox=(item.extent[0] - factor,
                                                      item.extent[1] - factor,
                                                      item.extent[2] + factor,
                                                      item.extent[3] + factor))
                        except:
                            pass
            self._sindex = qi
        elif HAS_SINDEX:
            #(xmin, ymin, xmax, ymax)
            if self.geometry_type.lower() == "point":
                stream = ((i, (item.extent.XMin - .01, item.extent.YMin - .01, item.extent.XMax + .01, item.extent.YMax + .01), idx) for i, (idx, item) in
                      enumerate(self.geometry.iteritems()) if
                      pd.notnull(item) and item)
            else:
                stream = ((i, (item.extent.XMin, item.extent.YMin, item.extent.XMax, item.extent.YMax), idx) for i, (idx, item) in
                          enumerate(self.geometry.iteritems()) if
                          pd.notnull(item) and item)
            try:
                self._sindex = SpatialIndex(stream)
            # What we really want here is an empty generator error, or
            # for the bulk loader to log that the generator was empty
            # and move on. See https://github.com/Toblerity/rtree/issues/20.
            except RTreeError:
                pass
    @property
    def sindex(self):
        if not self._sindex_valid:
            self._generate_sindex()
            self._sindex_valid = True
        return self._sindex
    #----------------------------------------------------------------------
    def _delete_index(self):
        """
        erases the spatial index and will be rebuilt
        """
        self._sindex = None
        self._sindex_valid = False
    #----------------------------------------------------------------------
    @property
    def JSON(self):
        """Returns an Esri JSON representation of the geometry as a string."""
        return _call_property(this=self, op="JSON")
    #----------------------------------------------------------------------
    @property
    def as_arcpy(self):
        """Returns an Esri ArcPy geometry in a Series"""

        return _call_property(this=self, op="as_arcpy", isGeoseries=False)
    #----------------------------------------------------------------------
    @property
    def as_shapely(self):
        """Returns a Shapely Geometry Objects in a Series"""
        return _call_property(this=self, op="as_shapely", isGeoseries=False)
    #----------------------------------------------------------------------
    @property
    def WKB(self):
        """Returns the well-known binary (WKB) representation for OGC
        geometry. It provides a portable representation of a geometry value
        as a contiguous stream of bytes."""
        return _call_property(this=self, op="WKB")
    #----------------------------------------------------------------------
    @property
    def WKT(self):
        """Returns the well-known text (WKT) representation for OGC
        geometry. It provides a portable representation of a geometry value
        as a text string."""
        return _call_property(this=self, op="WKT")
    #----------------------------------------------------------------------
    @property
    def area(self):
        """The area of a polygon feature. Empty for all other feature
        types."""
        return _call_property(this=self, op="area")
    #----------------------------------------------------------------------
    @property
    def centroid(self):
        """The true centroid if it is within or on the feature; otherwise,
        the label point is returned. Returns a point object."""
        return _call_property(this=self, op="centroid", isGeoseries=True)
    #----------------------------------------------------------------------
    @property
    def extent(self):
        """the extent of the geometry"""
        return _call_property(this=self, op="extent", isGeoseries=True)
    #----------------------------------------------------------------------
    @property
    def first_point(self):
        """The first coordinate point of the geometry."""
        return _call_property(this=self, op="first_point", isGeoseries=True)
    #----------------------------------------------------------------------
    @property
    def hull_rectangle(self):
        """A space-delimited string of the coordinate pairs of the convex
        hull rectangle."""
        return _call_property(this=self, op="hull_rectangle")
    #----------------------------------------------------------------------
    @property
    def is_multipart(self):
        """True, if the number of parts for the geometry is more than 1"""
        return _call_property(this=self, op="is_multipart")
    #----------------------------------------------------------------------
    @property
    def label_point(self):
        """
        The point at which the label is located. The labelPoint is always
        located within or on a feature.
        """
        return _call_property(this=self, op="label_point", isGeoseries=True)
    #----------------------------------------------------------------------
    @property
    def last_point(self):
        """
        The last coordinate of the feature.
        """
        return _call_property(this=self, op="last_point", isGeoseries=True)
    #----------------------------------------------------------------------
    @property
    def length(self):
        """The length of the linear feature. Zero for point and multipoint
        feature types."""
        return _call_property(this=self, op="length")
    #----------------------------------------------------------------------
    @property
    def length3D(self):
        """The 3D length of the linear feature. Zero for point and
        multipoint feature types."""
        return _call_property(this=self, op="length3D")
    #----------------------------------------------------------------------
    @property
    def part_count(self):
        """The number of geometry parts for the feature."""
        return _call_property(this=self, op="part_count")
    #----------------------------------------------------------------------
    @property
    def point_count(self):
        """The total number of points for the feature."""
        return _call_property(this=self, op="point_count")
    #----------------------------------------------------------------------
    @property
    def spatial_reference(self):
        """The spatial reference of the geometry."""
        return _call_property(this=self, op="spatial_reference")
    #----------------------------------------------------------------------
    @property
    def true_centroid(self):
        """The center of gravity for a feature."""
        return _call_property(this=self, op="true_centroid", isGeoseries=True)
    #----------------------------------------------------------------------
    @property
    def geometry_type(self):
        """The geometry type: polygon, polyline, point, multipoint, multipatch, dimension, or annotation"""
        from . import GeoSeries

        if isinstance(self, GeoSeries) and \
           isinstance(self.first_valid_index(), integer_types) and \
           self.first_valid_index() >= 0:
            return self[self.first_valid_index()].type.lower()
        if 'SHAPE' in self.columns and \
           len(self['SHAPE']) > 0:
            val = self['SHAPE'].loc[self['SHAPE'].first_valid_index()]
            if HASARCPY and \
               isinstance(val, (arcpy.Point, arcpy.PointGeometry)):
                return "point"
            elif HASARCPY and \
                 isinstance(val, (arcpy.Polygon)):
                return 'polygon'
            elif HASARCPY and \
                 isinstance(val, (arcpy.Polyline)):
                return 'polyline'
            elif HASARCPY and \
                 isinstance(val, (arcpy.Multipatch)):
                return 'multipatch'
            elif HASARCPY and \
                 isinstance(val, (arcpy.Multipoint)):
                return 'multipoint'
            elif HASARCPY and \
                 isinstance(val, (arcpy.Dimension)):
                return 'dimension'
            elif HASARCPY and \
                 isinstance(val, (arcpy.Annotation)):
                return 'annotation'
            elif hasattr(val, "geometry_type"):
                return getattr(val, 'type')
        return 'unknown'
    #----------------------------------------------------------------------
    @property
    def is_empty(self):
        """Return True for each empty geometry, False for non-empty"""
        return _call_property(self, 'point_count', null_value=False) == 0
    #
    #  Geometry Methods
    #
    #----------------------------------------------------------------------
    def angle_distance_to(self, second_geometry, method="GEODESIC"):
        """
        Returns a tuple of angle and distance to another point using a
        measurement type.

        Paramters:
         :second_geometry: - a second geometry
         :method: - PLANAR measurements reflect the projection of geographic
          data onto the 2D surface (in other words, they will not take into
          account the curvature of the earth). GEODESIC, GREAT_ELLIPTIC,
          LOXODROME, and PRESERVE_SHAPE measurement types may be chosen as
          an alternative, if desired.
        """
        other = second_geometry
        if isinstance(other, BaseSpatialPandas):
            raise ValueError("Input second_geometry must be of type: "\
                             "A arcgis.types.Geometry not %s" % type(second_geometry))
        return _call_function(this=self,
                              op="angle_distance_to",
                              second_geometry=other,
                              method=method)
    #----------------------------------------------------------------------
    def boundary(self):
        """
        Constructs the boundary of the geometry.

        """
        return _call_function(this=self, op="boundary", isGeoseries=True)
    #----------------------------------------------------------------------
    def buffer(self, distance):
        """
        Constructs a polygon at a specified distance from the geometry.

        Parameters:
         :distance: - length in current projection.  Only polygon accept
          negative values.
        """
        return _call_function(this=self, op="buffer",
                              distance=distance,
                              isGeoseries=True)
    #----------------------------------------------------------------------
    def clip(self, envelope):
        """
        Constructs the intersection of the geometry and the specified extent.

        Parameters:
         :envelope: - arcpy.Extent object
        """
        return _call_function(this=self, op='clip',
                              isGeoseries=True, envelope=envelope)
    #----------------------------------------------------------------------
    def contains(self, second_geometry, relation=None):
        """
        Indicates if the base geometry contains the comparison geometry.


        Paramters:
         :second_geometry: - a second geometry
        """
        return  _call_function(this=self,
                              op='contains',
                              isGeoseries=False,
                              relation=relation,
                              second_geometry=second_geometry)
    #----------------------------------------------------------------------
    def convex_hull(self):
        """
        Constructs the geometry that is the minimal bounding polygon such
        that all outer angles are convex.
        """
        return _call_function(this=self, op="convex_hull", isGeoseries=True)
    #----------------------------------------------------------------------
    def crosses(self, second_geometry):
        """
        Indicates if the two geometries intersect in a geometry of a lesser
        shape type.


        Paramters:
         :second_geometry: - a second geometry
        """
        return _call_function(this=self,
                              op='crosses',
                              isGeoseries=False,
                              second_geometry=second_geometry)
    #----------------------------------------------------------------------
    def cut(self, cutter):
        """
        Splits this geometry into a part left of the cutting polyline, and
        a part right of it.

        Parameters:
         :cutter: - The cutting polyline geometry.
        """
        if isinstance(cutter, GEOTYPES):
            return _call_function(this=self,
                                  op='cut',
                                  second_geometry=cutter,
                                  isGeoseries=True)
        else:
            raise ValueError("Invalid geometry given, must be of type %s" % GEOTYPES)
    #----------------------------------------------------------------------
    def densify(self, method, distance, deviation):
        """
        Creates a new geometry with added vertices

        Parameters:
         :method: - The type of densification, DISTANCE, ANGLE, or GEODESIC
         :distance: - The maximum distance between vertices. The actual
          distance between vertices will usually be less than the maximum
          distance as new vertices will be evenly distributed along the
          original segment. If using a type of DISTANCE or ANGLE, the
          distance is measured in the units of the geometry's spatial
          reference. If using a type of GEODESIC, the distance is measured
          in meters.
         :deviation: - Densify uses straight lines to approximate curves.
          You use deviation to control the accuracy of this approximation.
          The deviation is the maximum distance between the new segment and
          the original curve. The smaller its value, the more segments will
          be required to approximate the curve.
        """
        return _call_function(this=self, op='densify', method=method,
                              distance=distance, deviation=deviation,
                              second_geometry=None, isGeoseries=True)
    #----------------------------------------------------------------------
    def difference(self, second_geometry):
        """
        Constructs the geometry that is composed only of the region unique
        to the base geometry but not part of the other geometry. The
        following illustration shows the results when the red polygon is the
        source geometry.

        Paramters:
         :second_geometry: - a second geometry
        """
        return _call_function(this=self, op='difference',
                              second_geometry=second_geometry, isGeoseries=True)
    #----------------------------------------------------------------------
    def disjoint(self, second_geometry):
        """
        Indicates if the base and comparison geometries share no points in
        common.

        Paramters:
         :second_geometry: - a second geometry
        """
        return _call_function(this=self, op='disjoint',
                              second_geometry=second_geometry)
    #----------------------------------------------------------------------
    def distance_to(self, second_geometry):
        """
        Returns the minimum distance between two geometries. If the
        geometries intersect, the minimum distance is 0.
        Both geometries must have the same projection.

        Paramters:
         :second_geometry: - a second geometry
        """
        return _call_function(this=self, op='distance_to',
                              second_geometry=second_geometry,
                              null_value=-1)
    #----------------------------------------------------------------------
    def equals(self, second_geometry):
        """
        Indicates if the base and comparison geometries are of the same
        shape type and define the same set of points in the plane. This is
        a 2D comparison only; M and Z values are ignored.
        Paramters:
         :second_geometry: - a second geometry
        """
        return _call_function(this=self, op='equals',
                              second_geometry=second_geometry,
                              null_value=False)
    #----------------------------------------------------------------------
    def generalize(self, max_offset):
        """
        Creates a new simplified geometry using a specified maximum offset
        tolerance.

        Parameters:
         :max_offset: - The maximum offset tolerance.
        """
        return _call_function(this=self, op='generalize',
                              max_offset=max_offset, isGeoseries=True)
    #----------------------------------------------------------------------
    def get_area(self, method, units=None):
        """
        Returns the area of the feature using a measurement type.

        Parameters:
         :method: - PLANAR measurements reflect the projection of
          geographic data onto the 2D surface (in other words, they will not
          take into account the curvature of the earth). GEODESIC,
          GREAT_ELLIPTIC, LOXODROME, and PRESERVE_SHAPE measurement types
          may be chosen as an alternative, if desired.
         :units: - Areal unit of measure keywords: ACRES | ARES | HECTARES
          | SQUARECENTIMETERS | SQUAREDECIMETERS | SQUAREINCHES | SQUAREFEET
          | SQUAREKILOMETERS | SQUAREMETERS | SQUAREMILES |
          SQUAREMILLIMETERS | SQUAREYARDS

        """
        return _call_function(this=self, op='get_area', method=method,
                              units=units,
                              second_geometry=None)
    #----------------------------------------------------------------------
    def get_length(self, method, units):
        """
        Returns the length of the feature using a measurement type.

        Parameters:
         :method: - PLANAR measurements reflect the projection of
          geographic data onto the 2D surface (in other words, they will not
          take into account the curvature of the earth). GEODESIC,
          GREAT_ELLIPTIC, LOXODROME, and PRESERVE_SHAPE measurement types
          may be chosen as an alternative, if desired.
         :units: - Linear unit of measure keywords: CENTIMETERS |
          DECIMETERS | FEET | INCHES | KILOMETERS | METERS | MILES |
          MILLIMETERS | NAUTICALMILES | YARDS

        """
        return _call_function(this=self, op='get_length',
                              method=method, units=units)
    #----------------------------------------------------------------------
    def get_part(self, index=None):
        """
        Returns an array of point objects for a particular part of geometry
        or an array containing a number of arrays, one for each part.

        Parameters:
         :index: - The index position of the geometry.
        """
        return _call_function(this=self, op='get_part',
                              index=index)
    #----------------------------------------------------------------------
    def intersect(self, second_geometry, dimension):
        """
        Constructs a geometry that is the geometric intersection of the two
        input geometries. Different dimension values can be used to create
        different shape types. The intersection of two geometries of the
        same shape type is a geometry containing only the regions of overlap
        between the original geometries.

        Paramters:
         :second_geometry: - a second geometry
         :dimension: - The topological dimension (shape type) of the
          resulting geometry.
            1  -A zero-dimensional geometry (point or multipoint).
            2  -A one-dimensional geometry (polyline).
            4  -A two-dimensional geometry (polygon).

        """
        other = second_geometry
        return _call_function(this=self, op='intersect',
                              second_geometry=other, isGeoseries=True,
                               dimension=dimension)
    #----------------------------------------------------------------------
    def measure_on_line(self, second_geometry, as_percentage=False):
        """
        Returns a measure from the start point of this line to the in_point.

        Paramters:
         :second_geometry: - a second geometry
         :as_percentage: - If False, the measure will be returned as a
          distance; if True, the measure will be returned as a percentage.
        """
        in_point = second_geometry
        return _call_function(this=self, op="measure_on_line",
                              in_point=in_point, as_percentage=as_percentage,
                              isGeoseries=False)
    #----------------------------------------------------------------------
    def overlaps(self, second_geometry):
        """
        Indicates if the intersection of the two geometries has the same
        shape type as one of the input geometries and is not equivalent to
        either of the input geometries.


        Paramters:
         :second_geometry: - a second geometry
        """
        return _call_function(this=self,
                              op='overlaps',
                              second_geometry=second_geometry,
                              isGeoseries=False)
    #----------------------------------------------------------------------
    def point_from_angle_and_distance(self, angle, distance, method='GEODESCIC'):
        """
        Returns a point at a given angle and distance in degrees and meters
        using the specified measurement type.

        Parameters:
         :angle: - The angle in degrees to the returned point.
         :distance: - The distance in meters to the returned point.
         :method: - PLANAR measurements reflect the projection of geographic
          data onto the 2D surface (in other words, they will not take into
          account the curvature of the earth). GEODESIC, GREAT_ELLIPTIC,
          LOXODROME, and PRESERVE_SHAPE measurement types may be chosen as
          an alternative, if desired.
        """
        return _call_function(this=self,
                              op='point_from_angle_and_distance',
                              angle=angle,
                              distance=distance,
                              method=method,
                              isGeoseries=True)
    #----------------------------------------------------------------------
    def coordinates(self):
        """
        returns the point coordinates of the geometry as a
        np.array object
        """
        return _call_function(this=self, op='coordinates')
    #----------------------------------------------------------------------
    def position_along_line(self, value, use_percentage=False):
        """
        Returns a point on a line at a specified distance from the beginning
        of the line.

        Parameters:
         :value: - The distance along the line.
         :use_percentage: - The distance may be specified as a fixed unit
          of measure or a ratio of the length of the line. If True, value
          is used as a percentage; if False, value is used as a distance.
          For percentages, the value should be expressed as a double from
          0.0 (0%) to 1.0 (100%).
        """
        return _call_function(this=self,
                              op='position_along_line',
                              second_geometry=None,
                              isGeoseries=True,
                              value=value,
                              use_percentage=use_percentage)
    #----------------------------------------------------------------------
    def project_as(self, spatial_reference, transformation_name=None):
        """
        Projects a geometry and optionally applies a geotransformation.


        Parameter:
         :spatial_reference: - The new spatial reference. This can be a
          SpatialReference object or the coordinate system name.
         :transformation_name: - The geotransformation name.
        """
        return _call_function(this=self,
                              op='project_as',
                              spatial_reference=spatial_reference,
                              transformation_name=transformation_name,
                              isGeoseries=True)
    #----------------------------------------------------------------------
    def query_point_and_distance(self, second_geometry,
                                 use_percentage=False):
        """
        Finds the point on the polyline nearest to the in_point and the
        distance between those points. Also returns information about the
        side of the line the in_point is on as well as the distance along
        the line where the nearest point occurs.

        Paramters:
         :second_geometry: - a second geometry
         :as_percentage: - if False, the measure will be returned as
          distance, True, measure will be a percentage
        """
        in_point = second_geometry
        return _call_function(this=self,
                              op='query_point_and_distance',
                              in_point=in_point,
                              use_percentage=use_percentage,
                              isGeoseries=False)
    #----------------------------------------------------------------------
    def segment_along_line(self, start_measure,
                           end_measure, use_percentage=False):
        """
        Returns a Polyline between start and end measures. Similar to
        Polyline.positionAlongLine but will return a polyline segment between
        two points on the polyline instead of a single point.

        Parameters:
         :start_measure: - The starting distance from the beginning of the
          line.
         :end_measure: - The ending distance from the beginning of the
          line.
         :use_percentage: - The start and end measures may be specified as
          fixed units or as a ratio. If True, start_measure and end_measure
          are used as a percentage; if False, start_measure and end_measure
          are used as a distance. For percentages, the measures should be
          expressed as a double from 0.0 (0 percent) to 1.0 (100 percent).
        """
        return _call_function(this=self,
                              op='segment_along_line',
                              start_measure=start_measure,
                              end_measure=end_measure,
                              use_percentage=use_percentage,
                              isGeoseries=True)
    #----------------------------------------------------------------------
    def snap_to_line(self, second_geometry):
        """
        Returns a new point based on in_point snapped to this geometry.

        Paramters:
         :second_geometry: - a second geometry
        """
        in_point = second_geometry
        return _call_function(this=self,
                              op='snap_to_line',
                              in_point=in_point,
                              isGeoseries=True)
    #----------------------------------------------------------------------
    def symmetric_difference(self, second_geometry):
        """
        Constructs the geometry that is the union of two geometries minus
        the instersection of those geometries.
        The two input geometries must be the same shape type.
        Parameters:
         :second_geometry: - a second geometry
        """
        return _call_function(this=self,
                              op='symmetric_difference',
                              second_geometry=second_geometry,
                              isGeoseries=True)

    #----------------------------------------------------------------------
    def touches(self, second_geometry):
        """
        Indicates if the boundaries of the geometries intersect.


        Paramters:
         :second_geometry: - a second geometry
        """
        return _call_function(this=self,
                              op='touches',
                              second_geometry=second_geometry,
                              isGeoseries=False)
    #----------------------------------------------------------------------
    def union(self, second_geometry):
        """
        Constructs the geometry that is the set-theoretic union of the input
        geometries.


        Paramters:
         :second_geometry: - a second geometry
        """
        return _call_function(this=self,
                              op='union',
                              isGeoseries=True,
                              second_geometry=second_geometry)
    #----------------------------------------------------------------------
    def within(self, second_geometry, relation=None):
        """
        Indicates if the base geometry is within the comparison geometry.
        Paramters:
         :second_geometry: - a second geometry
         :relation: - The spatial relationship type.
          BOUNDARY  - Relationship has no restrictions for interiors or boundaries.
          CLEMENTINI  - Interiors of geometries must intersect. Specifying
           CLEMENTINI is equivalent to specifying None. This is the
           default.
          PROPER  - Boundaries of geometries must not intersect.

        """
        return _call_function(this=self,
                              op='within',
                              relation=relation,
                              second_geometry=second_geometry,
                              isGeoseries=False)
    #----------------------------------------------------------------------
    @property
    def bounds(self):
        """Return a DataFrame of minx, miny, maxx, maxy values of geometry objects"""
        if HASARCPY:
            x = self.geometry.extent
            x = np.array(x.tolist())
            return DataFrame(x,
                             columns=['xmin', 'ymin', 'xmax', 'ymax'],
                             index=self.index)
    #----------------------------------------------------------------------
    @property
    def series_extent(self):
        """Return a single bounding box (xmin, ymin, xmax, ymax) for all geometries

        This is a shortcut for calculating the min/max x and y bounds individually.
        """
        if HASARCPY:
            try:
                b = self.bounds
                return (b['xmin'].min(),
                        b['ymin'].min(),
                        b['xmax'].max(),
                        b['ymax'].max())
            except:
                return None
        else:
            array = np.array(self.extent.tolist())
            xmin = array[:,0].min()
            ymin = array[:,1].min()
            xmax = array[:,2].max()
            ymax = array[:,3].max()
            return (xmin, ymin, xmax, ymax)

