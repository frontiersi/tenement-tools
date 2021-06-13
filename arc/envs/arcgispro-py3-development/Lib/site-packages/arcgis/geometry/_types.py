"""
New Geometries Classes
"""
import copy
import json
import ujson as _ujson
try:
    import numpy as np
except ImportError as e:
    pass
from six import add_metaclass
from functools import partial

_number_type = (int, float)
_empty_value = [None, "NaN"]


def _is_valid(value):
    """checks if the value is valid"""

    if not isinstance(
        value.get('spatialReference', None),
        (dict, SpatialReference)
    ):
        return False

    if isinstance(value, Point):
        if hasattr(value, 'x') and \
           hasattr(value, 'y'):
            return True
        elif 'x' in value and \
             (value['x'] in _empty_value):
            return True
        return False
    elif isinstance(value, Envelope):
        if all(
            isinstance(getattr(value, extent, None), _number_type)
            for extent in ('xmin', 'ymin', 'xmax', 'ymax')
        ):
            return True
        elif hasattr(value, "xmin") and \
             (value.xmin in _empty_value):
            return True

        return False
    elif isinstance(value, (MultiPoint,
                            Polygon,
                            Polyline)):
        if 'paths' in value:
            if len(value['paths']) == 0:
                return True
            return _is_line(coords=value['paths'])
        elif 'rings' in value:
            if len(value['rings']) == 0:
                return True
            return _is_polygon(coords=value['rings'])
        elif 'points' in value:
            if len(value['points']) == 0:
                return True
            return _is_point(coords=value['points'])

    return False


def _is_polygon(coords):

    for coord in coords:
        if len(coord) < 4:
            return False
        if not _is_line(coord):
            return False
        if coord[0] != coord[-1]:
            return False

    return True


def _is_line(coords):
    """
    checks to see if the line has at
    least 2 points in the list
    """
    list_types = (list, tuple, set)
    if isinstance(coords, list_types) and \
       len(coords) > 0:
        return all(_is_point(elem) for elem in coords)

    return True


def _is_point(coords):
    """
    checks to see if the point has at
    least 2 coordinates in the list
    """
    valid = False
    if isinstance(coords, (list, tuple)) and len(coords) > 1:
        for coord in coords:
            if not isinstance(coord, _number_type):
                if not _is_point(coord):
                    return False
            valid = True

    return valid


def _geojson_type_to_esri_type(type_):
    mapping = {
        'LineString': Polyline,
        'MultiLineString': Polyline,
        'Polygon': Polygon,
        'MultiPolygon': Polygon,
        'Point': Point,
        'MultiPoint': MultiPoint
    }
    if mapping.get(type_):
        return mapping[type_]
    else:
        raise ValueError("Unknown GeoJSON Geometry type: {}".format(type_))


class BaseGeometry(dict):
    _ao = None
    _type = None
    _HASARCPY = None
    _HASSHAPELY = None
    _class_attributes = {'_ao', '_type', '_HASARCPY', '_HASSHAPELY', '_ipython_canary_method_should_not_exist_'}

    def __init__(self, iterable=None):
        if iterable is None:
            iterable = {}
        self.update(iterable)

    def is_valid(self):
        return _is_valid(self)

    def _check_geometry_engine(self):
        self._HASARCPY = True
        try:
            import arcpy
        except:
            self._HASARCPY = False

        self._HASSHAPELY = True
        try:
            import shapely
        except:
            self._HASSHAPELY = False

        return self._HASARCPY, self._HASSHAPELY

    def __setattr__(self, key, value):
        """sets the attribute"""
        if key in self._class_attributes:
            super(BaseGeometry, self).__setattr__(key, value)
        else:
            self[key] = value
            self._ao = None

    def __setattribute__(self, key, value):
        if key in self._class_attributes:
            super(BaseGeometry, self).__setattr__(key, value)
        else:
            self[key] = value
            self._ao = None

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        self._ao = None

    def __getattribute__(self, name):
        return super(BaseGeometry, self).__getattribute__(name)

    def __getattr__(self, name):
        try:
            if name in self._class_attributes:
                return super(BaseGeometry, self).__getattr__(name)
            return self.__getitem__(name)
        except:
            raise AttributeError("'%s' object has no attribute '%s'" % (type(self).__name__, name))

    def __getitem__(self, k):
        return dict.__getitem__(self, k)


class GeometryFactory(type):
    """
    Creates the Geometry Objects Based on JSON
    """

    @staticmethod
    def _from_wkb(iterable):
        _HASARCPY = True
        try:
            import arcpy
        except:
            _HASARCPY = False
        if _HASARCPY:
            return _ujson.loads(arcpy.FromWKB(iterable).JSON)
        return {}

    @staticmethod
    def _from_wkt(iterable):
        _HASARCPY = True
        try:
            import arcpy
        except:
            _HASARCPY = False
        if _HASARCPY:
            if "SRID=" in iterable:
                wkid, iterable = iterable.split(";")
                geom = _ujson.loads(arcpy.FromWKT(iterable).JSON)
                geom['spatialReference'] = {'wkid' : int(wkid.replace("SRID=",""))}
                return geom
            return _ujson.loads(arcpy.FromWKT(iterable).JSON)
        return {}

    @staticmethod
    def _from_gj(iterable):
        _HASARCPY = True
        try:
            import arcpy
        except:
            _HASARCPY = False
        if _HASARCPY:
            gj = _ujson.loads(arcpy.AsShape(iterable, False).JSON)
            gj['spatialReference']['wkid'] = 4326
            return gj
        else:
            cls = _geojson_type_to_esri_type(iterable['type'])
            return cls._from_geojson(iterable)

    def __call__(cls, iterable=None, **kwargs):
        if iterable is None:
            iterable = {}

        if iterable:
            # WKB
            if isinstance(iterable, (bytearray, bytes)):
                iterable = GeometryFactory._from_wkb(iterable)
            elif hasattr(iterable, "JSON"):
                iterable = _ujson.loads(getattr(iterable, "JSON"))
            elif 'coordinates' in iterable:
                iterable = GeometryFactory._from_gj(iterable)
            elif hasattr(iterable, "exportToString"):
                iterable = {'wkt': iterable.exportToString()}
            elif isinstance(iterable, str) and \
                    "{" in iterable:
                iterable = _ujson.loads(iterable)
            elif isinstance(iterable, str):  # WKT
                iterable = GeometryFactory._from_wkt(iterable)

            if 'x' in iterable:
                cls = Point
            elif 'rings' in iterable or "curveRings" in iterable:
                cls = Polygon
            elif 'curvePaths' in iterable or 'paths' in iterable:
                cls = Polyline
            elif 'points' in iterable:
                cls = MultiPoint
            elif 'xmin' in iterable:
                cls = Envelope
            elif 'wkid' in iterable or 'wkt' in iterable:
                return SpatialReference(iterable=iterable)
            elif isinstance(iterable, list):
                return Point({'x': iterable[0], 'y': iterable[1],
                              'spatialReference':
                                  {'wkid': kwargs.pop('wkid', 4326)}})
            else:
                cls = Geometry
        return type.__call__(cls, iterable, **kwargs)


@add_metaclass(GeometryFactory)
class Geometry(BaseGeometry):
    """
    The base class for all geometries.

    You can create a Geometry even when you don't know the exact type. The Geometry constructor is able
    to figure out the geometry type and returns the correct type as the example below demonstrates:

    .. code-block:: python

        geom = Geometry({
          "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
                      [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
                      [-97.06326,32.759]]],
          "spatialReference" : {"wkid" : 4326}
        })
        print (geom.type) # POLYGON
        print (isinstance(geom, Polygon) # True

    """

    def __init__(self, iterable=None, **kwargs):
        if iterable is None:
            iterable = ()
        super(Geometry, self).__init__(iterable, **kwargs)

    @property
    def __geo_interface__(self):
        """
        Converts an ESRI JSON to GeoJSON

        :returns: string
        """
        _HASARCPY, _HASSHAPELY = self._check_geometry_engine()
        if _HASARCPY:
            import arcpy
            if isinstance(self.as_arcpy, arcpy.Point):
                return arcpy.PointGeometry(self.as_arcpy).__geo_interface__
            else:
                return self.as_arcpy.__geo_interface__
        else:
            if isinstance(self, Point):
                return {'type': 'Point', 'coordinates': (self.x,
                                                         self.y)}
            elif isinstance(self, Polygon):
                col = []
                for part in self['rings']:
                    col.append([tuple(pt) for pt in part])
                return {
                    'coordinates': [col],
                    'type': 'MultiPolygon'
                }
            elif isinstance(self, Polyline):
                return {'type': 'MultiLineString', 'coordinates': [[((pt[0], pt[1]) if pt else None)
                                                                    for pt in part]
                                                                   for part in self['paths']]}
            elif isinstance(self, MultiPoint):
                return {'type': 'Multipoint', 'coordinates': [(pt[0], pt[1]) for pt in self['points']]}

            from arcgis._impl.common._arcgis2geojson import arcgis2geojson
            return arcgis2geojson(arcgis=self)

    def __iter__(self):
        """
        Iterator for the Geometry
        """
        if isinstance(self, Polygon):
            avgs = []
            shape = 2
            for ring in self['rings']:
                np_array_ring = np.array(ring)
                shape = np_array_ring.shape[1]
                avgs.append([np_array_ring[:, 0].mean(), np_array_ring[:, 1].mean()])

            avgs = np.array(avgs)
            res = []
            if shape == 2:
                res = [avgs[:, 0].mean(),
                       avgs[:, 1].mean()]
            elif shape > 2:
                res = [avgs[:, 0].mean(),
                       avgs[:, 1].mean(),
                       avgs[:, 2].mean()]
            for a in res:
                yield a
                del a
        elif isinstance(self, Polyline):
            avgs = []
            shape = 2
            for ring in self['paths']:
                np_array_ring = np.array(ring)
                shape = np_array_ring.shape[1]
                avgs.append([np_array_ring[:, 0].mean(), np_array_ring[:,1].mean()])
            avgs = np.array(avgs)
            res = []
            if shape == 2:
                res = [avgs[:, 0].mean(),
                       avgs[:, 1].mean()]
            elif shape > 2:
                res = [avgs[:, 0].mean(),
                       avgs[:, 1].mean(),
                       avgs[:, 2].mean()]
            for a in res:
                yield a
                del a
        elif isinstance(self, MultiPoint):
            a = np.array(self['points'])
            if a.shape[1] == 2:
                for i in [a[:, 0].mean(),
                          a[:, 1].mean()]:
                    yield i
            elif a.shape[1] >= 3: #has z
                for i in [a[:, 0].mean(),
                          a[:, 1].mean(),
                          a[:, 2].mean()]:
                    yield i
        elif isinstance(self, Point):
            keys = ['x', 'y', 'z']
            for k in keys:
                if k in self:
                    yield self[k]
                del k
        elif isinstance(self, Envelope):
            for i in [(self['xmin'] + self['xmax'])/2,
                      (self['ymin'] + self['ymax'])/2]:
                yield i

    def _repr_svg_(self):
        """SVG representation for iPython notebook"""
        svg_top = '<svg xmlns="http://www.w3.org/2000/svg" ' \
            'xmlns:xlink="http://www.w3.org/1999/xlink" '
        if self.is_empty:
            return svg_top + '/>'
        else:

            # Establish SVG canvas that will fit all the data + small space
            xmin, ymin, xmax, ymax = self.extent
            # Expand bounds by a fraction of the data ranges
            expand = 0.04  # or 4%, same as R plots
            widest_part = max([xmax - xmin, ymax - ymin])
            expand_amount = widest_part * expand

            if xmin == xmax and ymin == ymax:
                # This is a point; buffer using an arbitrary size
                try:
                    xmin, ymin, xmax, ymax = self.buffer(1).extent
                except:
                    xmin -= expand_amount
                    ymin -= expand_amount
                    xmax += expand_amount
                    ymax += expand_amount
            else:
                xmin -= expand_amount
                ymin -= expand_amount
                xmax += expand_amount
                ymax += expand_amount

            dx = xmax - xmin
            dy = ymax - ymin
            width = min([max([100., dx]), 300])
            height = min([max([100., dy]), 300])

            try:
                scale_factor = max([dx, dy]) / max([width, height])
            except ZeroDivisionError:
                scale_factor = 1.

            view_box = "{0} {1} {2} {3}".format(xmin, ymin, dx, dy)
            transform = "matrix(1,0,0,-1,0,{0})".format(ymax + ymin)
            return svg_top + (
                'width="{1}" height="{2}" viewBox="{0}" '
                'preserveAspectRatio="xMinYMin meet">'
                '<g transform="{3}">{4}</g></svg>'
                ).format(view_box, width, height, transform,
                         self.svg(scale_factor))

    @property
    def as_arcpy(self):
        """
        Returns the Geometry as an ArcPy Geometry.

        If `ArcPy` is not installed, none is returned.

        **Requires ArcPy**

        :returns: arcpy.Geometry

        """
        _HASARCPY, _HASSHAPELY = self._check_geometry_engine()
        if self._ao is not None or not _HASARCPY:
            return self._ao

        if _HASARCPY:
            import arcpy

        if isinstance(self, (Point, MultiPoint, Polygon, Polyline)):
            self._ao = arcpy.AsShape(
                json.dumps(dict(self)),
                True)
        elif isinstance(self, SpatialReference):
            if 'wkid' in self:
                self._ao = arcpy.SpatialReference(self['wkid'])
            elif 'wkt' in self:
                self._ao = arcpy.SpatialReference(self['wkt'])
            else:
                raise ValueError("Invalid SpatialReference")
        elif isinstance(self, Envelope):
            return arcpy.Extent(XMin=self['xmin'],
                                YMin=self['ymin'],
                                XMax=self['xmax'],
                                YMax=self['ymax'])
        return self._ao

    def _wkt(obj, fmt='%.16f'):
        """converts an arcgis.Geometry to WKT"""
        if isinstance(obj, Point):
            coords = [obj['x'], obj['y']]
            if 'z' in obj:
                coords.append(obj['z'])
            return "POINT (%s)" % ' '.join(fmt % c for c in coords)
        elif isinstance(obj, Polygon):
            coords = obj['rings']
            pt2 = []
            b = "MULTIPOLYGON (%s)"
            for part in coords:
                c2 = []
                for c in part:
                    c2.append("(%s,  %s)" % (fmt % c[0], fmt % c[1]))
                j = "(%s)" % ", ".join(c2)
                pt2.append(j)
            b = b % ", ".join(pt2)
            return b
        elif isinstance(obj, Polyline):
            coords = obj['paths']
            pt2 = []
            b = "MULTILINESTRING (%s)"
            for part in coords:
                c2 = []
                for c in part:
                    c2.append("(%s,  %s)" % (fmt % c[0], fmt % c[1]))
                j = "(%s)" % ", ".join(c2)
                pt2.append(j)
            b = b % ", ".join(pt2)
            return b
        elif isinstance(obj, MultiPoint):
            coords = obj['points']
            b = "MULTIPOINT (%s)"
            c2 = []
            for c in coords:
                c2.append("(%s,  %s)" % (fmt % c[0], fmt % c[1]))
            return b % ", ".join(c2)
        return ""

    @property
    def geoextent(self):
        """
        Returns the current feature's extent

        >>> g = Geometry({...})
        >>> g.geoextent
        (1,2,3,4)


        :return: tuple
        """
        _HASARCPY, _HASSHAPELY = self._check_geometry_engine()

        if not hasattr(self, 'type'):
            return None

        a = None
        if str(self.type).upper() == "POLYGON":
            if 'rings' in self:
                a = self['rings']
            elif 'curveRings' in self:
                if not _HASARCPY:
                    raise Exception("Cannot calculate the geoextent with curves without ArcPy.")
                return (self.as_arcpy.extent.XMin,
                        self.as_arcpy.extent.YMin,
                        self.as_arcpy.extent.XMax,
                        self.as_arcpy.extent.YMax)
        elif str(self.type).upper() == "POLYLINE":
            if 'paths' in self:
                a = self['paths']
            elif "curvePaths" in self:
                if not _HASARCPY:
                    raise Exception("Cannot calculate the geoextent with curves without ArcPy.")
                return (self.as_arcpy.extent.XMin,
                        self.as_arcpy.extent.YMin,
                        self.as_arcpy.extent.XMax,
                        self.as_arcpy.extent.YMax)
        elif str(self.type).upper() == "MULTIPOINT":
            a = np.array(self['points'])
            x_max = max(a[:, 0])
            x_min = min(a[:, 0])
            y_min = min(a[:, 1])
            y_max = max(a[:, 1])
            return x_min, y_min, x_max, y_max
        elif str(self.type).upper() == "POINT":
            return self['x'], self['y'], self['x'],  self['y']
        elif str(self.type).upper() == "ENVELOPE":
            return tuple(self.coordinates().tolist())
        else:
            return None

        if a is None or len(a) == 0:
            return None

        if len(a) == 1: # single part
            x_max = max(a[0], key=lambda x: x[0])[0]
            x_min = min(a[0], key=lambda x: x[0])[0]
            y_max = max(a[0], key=lambda x: x[1])[1]
            y_min = min(a[0], key=lambda x: x[1])[1]
            return x_min, y_min, x_max, y_max
        else:
            if 'points' in a:
                a = a['points']
            elif 'coordinates' in a:
                a = a['coordinates']
            xs = []
            ys = []
            for pt in a: # multiple part geometry
                x_max = max(pt, key=lambda x: x[0])[0]
                x_min = min(pt, key=lambda x: x[0])[0]
                y_max = max(pt, key=lambda x: x[1])[1]
                y_min = min(pt, key=lambda x: x[1])[1]
                xs.append(x_max)
                xs.append(x_min)
                ys.append(y_max)
                ys.append(y_min)
                del pt
            return min(xs), min(ys), max(xs), max(ys)

    @property
    def envelope(self):
        """Returns the geoextent as an Envelope object"""
        env_dict = {'xmin': self.geoextent[0],
                    'ymin': self.geoextent[1],
                    'xmax': self.geoextent[2],
                    'ymax': self.geoextent[3],
                    'spatialReference': self.spatial_reference}

        return Envelope(env_dict)

    def skew(self, x_angle=0,
             y_angle=0, inplace=False):
        """
        Create a skew transform along one or both axes.

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        x_angle             optional Float. Angle to skew in the x coordinate
        ---------------     --------------------------------------------------------------------
        y_angle             Optional Float. Angle to skew in the y coordinate
        ---------------     --------------------------------------------------------------------
        inplace             Optional Boolean. If True, the value is updated in the object, False
                            creates a new object
        ===============     ====================================================================


        :return: Geometry

        """
        from .affine import skew
        s = skew(geom=copy.deepcopy(self), x_angle=x_angle,
                    y_angle=y_angle)
        if inplace:
            self.update(s)
        return s

    def rotate(self, theta,
               inplace=False):
        """
        Rotates a geometry counter-clockwise by a given angle.


        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        theta               Required Float. The rotation angle.
        ---------------     --------------------------------------------------------------------
        inplace             Optional Boolean. If True, the value is updated in the object, False
                            creates a new object
        ===============     ====================================================================


        :return: Geometry

        """
        from .affine import rotate

        r = rotate(copy.deepcopy(self), theta)
        if inplace:
            self.update(r)
        return r

    def scale(self, x_scale=1, y_scale=1, inplace=False):
        """
        Scales in either the x,y or both directions


        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        x_scale             Optional Float. The x-scale factor.
        ---------------     --------------------------------------------------------------------
        y_scale             Optional Float. The y-scale factor.
        ---------------     --------------------------------------------------------------------
        inplace             Optional Boolean. If True, the value is updated in the object, False
                            creates a new object
        ===============     ====================================================================


        :return: Geometry

        """
        from .affine import scale
        import copy

        g = copy.copy(self)
        s = scale(g, *(x_scale, y_scale))
        if inplace:
            self.update(s)
        return s

    def translate(self, x_offset=0,
                  y_offset=0, inplace=False):
        """
        moves a geometry in a given x and y distance


        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        x_offset            Optional Float. Translation x offset
        ---------------     --------------------------------------------------------------------
        y_offset            Optional Float. Translation y offset
        ---------------     --------------------------------------------------------------------
        inplace             Optional Boolean. If False, updates the existing Geometry,else it
                            creates a new Geometry object
        ===============     ====================================================================


        :return: Geometry

        """
        from .affine import translate
        t = translate(copy.deepcopy(self), x_offset, y_offset)
        if inplace:
            self.update(t)
        return t

    @property
    def is_empty(self):
        """boolean value that determines if the geometry is empty or not"""
        if isinstance(self, Point):
            return False
        elif isinstance(self, Polygon):
            if 'rings' in self:
                return len(self['rings']) == 0
            elif 'curveRings' in self:
                return len(self['curveRings']) == 0
        elif isinstance(self, Polyline):
            return len(self['paths']) == 0
        elif isinstance(self, MultiPoint):
            return len(self['points']) == 0
        return True

    @property
    def as_shapely(self):
        """returns a shapely geometry object"""
        _, _HASSHAPELY = self._check_geometry_engine()
        if _HASSHAPELY:
            if isinstance(self,(Point, Polygon, Polyline, MultiPoint)):
                from shapely.geometry import shape
                if 'curvePaths' in self or 'curveRings' in self:
                    return {}
                return shape(self.__geo_interface__)
        return None

    @property
    def JSON(self):
        """
        Returns an Esri JSON representation of the geometry as a string.

        :return: string
        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY:
            import arcpy
        if HASARCPY and \
           isinstance(self.as_arcpy, arcpy.Geometry):
            return getattr(self.as_arcpy, "JSON", None)
        elif HASSHAPELY:
            try:
                return json.dumps(self.as_shapely.__geo_interface__)
            except:
                return json.dumps(self)

        return json.dumps(self)
    #----------------------------------------------------------------------
    @classmethod
    def from_shapely(cls, shapely_geometry, spatial_reference=None):
        """
        Creates a Python API Geometry object from a Shapely geometry object.


        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        shapely_geometry    Required Shapely Geometry
                            Single instance of Shapely Geometry to be converted to ArcGIS
                            Python API geometry instance.
        ---------------     --------------------------------------------------------------------
        spatial_reference   Optional SpatialReference
                            Defines the spatial reference for the output geometry.
        ===============     ====================================================================

        :return: arcgis.geometry.Geometry

        .. code-block:: python

            # Usage Example: importing shapely geometry object and setting spatial reference to WGS84

            Geometry.from_shapely(
                shapely_geometry=shapely_geometry_object,
                spatial_reference={'wkid': 4326}
            )

        """
        _HASSHAPELY = True
        try:
            import shapely
        except:
            _HASSHAPELY = False
        if _HASSHAPELY:
            gj = shapely_geometry.__geo_interface__
            geom_cls = _geojson_type_to_esri_type(gj['type'])

            if spatial_reference:
                geometry = geom_cls._from_geojson(gj,sr=spatial_reference)
            else:
                geometry = geom_cls._from_geojson(gj)

            return geometry
        else:
            raise ValueError('Shapely is required to execute from_shapely.')
    #----------------------------------------------------------------------
    @property
    def EWKT(self):
        """
        Returns the extended well-known text (EWKT) representation for OGC geometry.
        It provides a portable representation of a geometry value as a text
        string.
        Any true curves in the geometry will be densified into approximate
        curves in the WKT string.

        :return: string
        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY and \
               isinstance(self, Envelope):
            try:
                p = getattr(self.as_arcpy, 'polygon', None)
                sr = self.spatial_reference.get('wkid', 4326)
                return f"SRID={sr};{p.WKT}"
            except:
                return None
        if HASARCPY:
            sr = self.spatial_reference.get('wkid', 4326)
            return f"SRID={sr};{getattr(self.as_arcpy, 'WKT', None)}"
        elif HASSHAPELY:
            try:
                sr = self.spatial_reference.get('wkid', 4326)
                return f"SRID={sr};{self.as_shapely.wkt}"
            except:
                sr = self.spatial_reference.get('wkid', 4326)
                return f"SRID={sr};{self._wkt(fmt='%.16f')}"
        else:
            sr = self.spatial_reference.get('wkid', 4326)
            return f"SRID={sr};{self._wkt(fmt='%.16f')}"
    #----------------------------------------------------------------------
    @property
    def WKT(self):
        """
        Returns the well-known text (WKT) representation for OGC geometry.
        It provides a portable representation of a geometry value as a text
        string.
        Any true curves in the geometry will be densified into approximate
        curves in the WKT string.

        :return: string
        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY and \
           isinstance(self, Envelope):
            try:
                p = getattr(self.as_arcpy, 'polygon', None)
                return p.WKT
            except:
                return None
        if HASARCPY:
            return getattr(self.as_arcpy, "WKT", None)
        elif HASSHAPELY:
            try:
                return self.as_shapely.wkt
            except:
                return self._wkt(fmt='%.16f')

        return self._wkt(fmt='%.16f')
    #----------------------------------------------------------------------
    @property
    def WKB(self):
        """
        Returns the well-known binary (WKB) representation for OGC geometry.
        It provides a portable representation of a geometry value as a
        contiguous stream of bytes.

        :return: bytes
        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY and \
           isinstance(self, Envelope):
            try:
                p = getattr(self.as_arcpy, 'polygon', None)
                return p.WKB
            except:
                return None
        if HASARCPY:
            try:
                return getattr(self.as_arcpy, "WKB", None)
            except:
                return None
        elif HASSHAPELY:
            try:
                return self.as_shapely.wkb
            except:
                return None
        return None
    #----------------------------------------------------------------------
    @property
    def area(self):
        """
        The area of a polygon feature. None for all other feature types.
        The area is in the units of the spatial reference.

        .. code-block:: python

            >>> geom = Geometry({
              "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
                          [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
                          [-97.06326,32.759]]],
              "spatialReference" : {"wkid" : 4326}
            })
            >>> geom.area
            -1.869999999973911e-06


        :return: float
        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY and \
           isinstance(self, Envelope):
            try:
                p = getattr(self.as_arcpy, 'polygon', None)
                return p.area
            except:
                return None
        if HASARCPY:
            return getattr(self.as_arcpy, "area", None)
        elif HASSHAPELY:
            return self.as_shapely.area
        elif isinstance(self, Polygon):
            return self._shoelace_area(parts=self['rings'])
        return None
    #----------------------------------------------------------------------
    def _shoelace_area(self, parts):
        """calculates the shoelace area"""
        area = 0.0
        area_parts = []
        for part in parts:
            n = len(part)
            for i in range(n):
                j = (i + 1) % n

                area += part[i][0] * part[j][1]
                area -= part[j][0] * part[i][1]

            area_parts.append(area / 2.0)
            area = 0.0
        return abs(sum(area_parts))
    #----------------------------------------------------------------------
    @property
    def centroid(self):
        """
        Returns the center of the geometry

        **Requires ArcPy or Shapely**

        .. code-block:: python

            >>> geom = Geometry({
              "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
                          [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
                          [-97.06326,32.759]]],
              "spatialReference" : {"wkid" : 4326}
            })
            >>> geom.centroid
            (-97.06258999999994, 32.754333333000034)


        :returns: tuple(x,y)
        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY and \
           isinstance(self, Envelope):
            try:
                p = getattr(self.as_arcpy, 'polygon', None)
                return p.centroid
            except:
                return None
        if HASARCPY:
            import arcpy
            if isinstance(self, Point):
                return tuple(self)
            else:
                g = getattr(self.as_arcpy, "centroid", None)
                if g is None:
                    return g
                return tuple(Geometry(
                    arcpy.PointGeometry(g,
                                        self.spatial_reference)
                ))
        elif HASSHAPELY:
            c = tuple(list(self.as_shapely.centroid.coords)[0])
            return c
        return
    #----------------------------------------------------------------------
    @property
    def extent(self):
        """
        The extent of the geometry as a tuple containing xmin, ymin, xmax, ymax

        **Requires ArcPy or Shapely**

        .. code-block:: python

            >>> geom = Geometry({
              "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
                          [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
                          [-97.06326,32.759]]],
              "spatialReference" : {"wkid" : 4326}
            })
            >>> geom.extent
            (-97.06326, 32.749, -97.06124, 32.837)

        :return: tuple
        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        ptX = []
        ptY = []
        if isinstance(self, Envelope):
            try:
                return tuple(self.coordinates.tolist())
            except:
                return None
        if HASARCPY:
            ext = getattr(self.as_arcpy, "extent", None)
            return ext.XMin, ext.YMin, ext.XMax, ext.YMax
        elif HASSHAPELY:
            return self.as_shapely.bounds
        elif isinstance(self, Polygon):
            for pts in self['rings']:
                for part in pts:
                    ptX.append(part[0])
                    ptY.append(part[1])
            return min(ptX), min(ptY), max(ptX), max(ptY)

        elif isinstance(self, Polyline):
            for pts in self['paths']:
                for part in pts:
                    ptX.append(part[0])
                    ptY.append(part[1])
            return min(ptX), min(ptY), max(ptX), max(ptY)
        elif isinstance(self, MultiPoint):
            ptX = [ pt['x'] for pt in self['points']]
            ptY = [ pt['y'] for pt in self['points']]
            return min(ptX), min(ptY), max(ptX), max(ptY)
        elif isinstance(self, Point):
            return self['x'], self['y'], self['x'], self['y']
        return
    #----------------------------------------------------------------------
    @property
    def first_point(self):
        """
        The first coordinate point of the geometry.

        .. code-block:: python

            >>> geom = Geometry({
              "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
                          [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
                          [-97.06326,32.759]]],
              "spatialReference" : {"wkid" : 4326}
            })
            >>> geom.first_point
            {'x': -97.06138, 'y': 32.837, 'spatialReference': {'wkid': 4326, 'latestWkid': 4326}}


        :return: arcgis.gis.Geometry
        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY and \
           isinstance(self, Envelope):
            try:
                return Geometry({'x' : self['XMin'],
                                 "y" : self['YMin'],
                                 'spatialReference' : self['spatialReference']})
            except:
                return None
        elif HASARCPY:
            import arcpy
            return Geometry(_ujson.loads(arcpy.PointGeometry(getattr(
                self.as_arcpy,
                "firstPoint",
                None), self.spatial_reference).JSON))
        elif isinstance(self, Point):
            return self
        elif isinstance(self, MultiPoint):
            if len(self['points']) == 0:
                return
            geom = self['points'][0]
            return Geometry(
                {'x': geom[0], 'y': geom[1],
                 'spatialReference' : {'wkid' : 4326}}
            )
        elif isinstance(self, Polygon):
            if len(self['rings']) == 0:
                return
            geom = self['rings'][0][0]
            return Geometry(
                {'x': geom[0], 'y': geom[1],
                 'spatialReference' : {'wkid' : 4326}}
            )
        elif isinstance(self, Polyline):
            if len(self['paths']) == 0:
                return
            geom = self['paths'][0][0]
            return Geometry(
                {'x': geom[0], 'y': geom[1],
                 'spatialReference' : {'wkid' : 4326}}
            )
        return
    #----------------------------------------------------------------------
    @property
    def has_z(self):
        """
        Determines if the geometry has a `Z` value.

        :returns: Boolean

        """
        return self.get("hasZ", False)
    #----------------------------------------------------------------------
    @property
    def has_m(self):
        """
        Determines if the geometry has a `M` value.

        :returns: Boolean

        """
        return self.get("hasM", False)
    #----------------------------------------------------------------------
    @property
    def hull_rectangle(self):
        """
        A space-delimited string of the coordinate pairs of the convex hull
        rectangle.

        **Requires ArcPy or Shapely**

        .. code-block:: python

            >>> geom = Geometry({
              "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
                          [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
                          [-97.06326,32.759]]],
              "spatialReference" : {"wkid" : 4326}
            })
            >>> geom.hull_rectangle
            '-97.06153 32.749 -97.0632940971127 32.7490060186843 -97.0629938635673 32.8370055061228 -97.0612297664546 32.8369994874385'

        :return: string
        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY and \
           isinstance(self, Envelope):
            return getattr(self.polygon.as_arcpy, 'hullRectangle', None)
        if HASARCPY:
            return getattr(self.as_arcpy, "hullRectangle", None)
        elif HASSHAPELY:
            return self.as_shapely.convex_hull
        return
    #----------------------------------------------------------------------
    @property
    def is_multipart(self):
        """
        True, if the number of parts for this geometry is more than one.

        **Requires ArcPy or Shapely**

        .. code-block:: python

            >>> geom = Geometry({
              "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
                          [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
                          [-97.06326,32.759]]],
              "spatialReference" : {"wkid" : 4326}
            })
            >>> geom.is_multipart
            True


        :return: boolean
        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY and \
           isinstance(self, Envelope):
            return False
        elif HASARCPY:
            return getattr(self.as_arcpy, "isMultipart", None)
        elif HASSHAPELY:
            if self.type.lower().find("multi") > -1:
                return True
            else:
                return False
        return
    #----------------------------------------------------------------------
    @property
    def label_point(self):
        """
        The point at which the label is located. The label_point is always
        located within or on a feature.

        **Requires ArcPy or Shapely**

        .. code-block:: python

            >>> geom = Geometry({
              "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
                          [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
                          [-97.06326,32.759]]],
              "spatialReference" : {"wkid" : 4326}
            })
            >>> geom.label_point
            {'x': -97.06258999999994, 'y': 32.754333333000034, 'spatialReference': {'wkid': 4326, 'latestWkid': 4326}}

        :returns: arcgis.geometry.Point

        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY and \
           isinstance(self, Envelope):
            return getattr(self.polygon.as_arcpy, "labelPoint", None)
        elif HASARCPY:
            import arcpy
            return Geometry(arcpy.PointGeometry(getattr(self.as_arcpy, "labelPoint", None),
                                                self.spatial_reference))

        return self.centroid
    #----------------------------------------------------------------------
    @property
    def last_point(self):
        """
        The last coordinate of the feature.

        .. code-block:: python

            >>> geom = Geometry({
              "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
                          [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
                          [-97.06326,32.759]]],
              "spatialReference" : {"wkid" : 4326}
            })
            >>> geom.last_point
            {'x': -97.06326, 'y': 32.759, 'spatialReference': {'wkid': 4326, 'latestWkid': 4326}}


        :returns: arcgis.geometry.Point
        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY and \
           isinstance(self, Envelope):
            return Geometry({'x' : self['XMax'], 'y' : self['YMax'],
                             'spatialReference' : self['spatialReference']})
        elif HASARCPY:
            import arcpy
            return Geometry(arcpy.PointGeometry(getattr(self.as_arcpy, "lastPoint", None),
                                                self.spatial_reference))
        elif isinstance(self, Point):
            return self
        elif isinstance(self, Polygon):
            if self['rings'] == 0:
                return
            geom = self['rings'][-1][-1]
            return Geometry(
                {'x': geom[0], 'y': geom[1],
                 'spatialReference' : {'wkid' : 4326}}
            )
        elif isinstance(self, Polyline):
            if self['paths'] == 0:
                return
            geom = self['paths'][-1][-1]
            return Geometry(
                {'x': geom[0], 'y': geom[1],
                 'spatialReference' : {'wkid' : 4326}}
            )
        return
    #----------------------------------------------------------------------
    @property
    def length(self):
        """
        The length of the linear feature. Zero for point and multipoint feature types.
        The length units is the same as the spatial reference.

        **Requires ArcPy or Shapely**

        .. code-block:: python

            >>> geom = Geometry({
              "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
                          [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
                          [-97.06326,32.759]]],
              "spatialReference" : {"wkid" : 4326}
            })
            >>> geom.length
            0.03033576008004027

        :return: float
        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY and \
           isinstance(self, Envelope):
            return getattr(self.polygon.as_arcpy, "length", None)
        elif HASARCPY:
            return getattr(self.as_arcpy, "length", None)
        elif HASSHAPELY:
            return self.as_shapely.length

        return None
    #----------------------------------------------------------------------
    @property
    def length3D(self):
        """
        The 3D length of the linear feature. Zero for point and multipoint
        feature types. The length units is the same as the spatial
        reference.

        **Requires ArcPy or Shapely**

        .. code-block:: python

            >>> geom = Geometry({
              "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
                          [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
                          [-97.06326,32.759]]],
              "spatialReference" : {"wkid" : 4326}
            })
            >>> geom.length3D
            0.03033576008004027

        :return: float
        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY and \
           isinstance(self, Envelope):
            return getattr(self.polygon.as_arcpy, "length3D", None)
        elif HASARCPY:
            return getattr(self.as_arcpy, "length3D", None)
        elif HASSHAPELY:
            return self.as_shapely.length

        return self.length
    #----------------------------------------------------------------------
    @property
    def part_count(self):
        """
        The number of geometry parts for the feature.


        .. code-block:: python

            >>> geom = Geometry({
              "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
                          [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
                          [-97.06326,32.759]]],
              "spatialReference" : {"wkid" : 4326}
            })
            >>> geom.part_count
            1

        :return: integer
        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY and \
           isinstance(self, Envelope):
            return 1
        elif HASARCPY:
            return getattr(self.as_arcpy, "partCount", None)
        elif isinstance(self, Polygon):
            return len(self['rings'])
        elif isinstance(self, Polyline):
            return len(self['paths'])
        elif isinstance(self, MultiPoint):
            return len(self['points'])
        elif isinstance(self, Point):
            return 1
        return
    #----------------------------------------------------------------------
    @property
    def point_count(self):
        """
        The total number of points for the feature.


        .. code-block:: python

            >>> geom = Geometry({
              "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
                          [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
                          [-97.06326,32.759]]],
              "spatialReference" : {"wkid" : 4326}
            })
            >>> geom.point_count
            9

        :return: Integer
        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if isinstance(self, Envelope):
            return 4
        elif HASARCPY:
            return getattr(self.as_arcpy, "pointCount", None)
        elif isinstance(self, Polygon):
            return sum([len(part) for part in self['rings']])
        elif isinstance(self, Polyline):
            return sum([len(part) for part in self['paths']])
        elif isinstance(self, MultiPoint):
            return sum([len(part) for part in self['points']])
        elif isinstance(self, Point):
            return 1
        return
    #----------------------------------------------------------------------
    @property
    def spatial_reference(self):
        """
        The spatial reference of the geometry.

        .. code-block:: python

            >>> geom = Geometry({
              "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
                          [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
                          [-97.06326,32.759]]],
              "spatialReference" : {"wkid" : 4326}
            })
            >>> geom.spatial_reference
            <SpatialReference Class>

        :return: arcgis.geometery.SpatialReference
        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY and \
           isinstance(self, Envelope):
            v = getattr(self.polygon.as_arcpy, "spatialReference", None)
            if v:
                return SpatialReference(v)
        elif HASARCPY:
            return SpatialReference(self['spatialReference'])
        if 'spatialReference' in self:
            return SpatialReference(self['spatialReference'])
        return None
    #----------------------------------------------------------------------
    @property
    def true_centroid(self):
        """
        The center of gravity for a feature.

        **Requires ArcPy or Shapely**

        .. code-block:: python

            >>> geom = Geometry({
              "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
                          [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
                          [-97.06326,32.759]]],
              "spatialReference" : {"wkid" : 4326}
            })
            >>> geom.true_centroid
            {'x': -97.06272135472369, 'y': 32.746201426025, 'spatialReference': {'wkid': 4326, 'latestWkid': 4326}}

        :returns: arcgis.geometry.Point
        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY:
            import arcpy
        if HASARCPY and \
           isinstance(self, Envelope):
            return Geometry(
                arcpy.PointGeometry(
                    getattr(self.polygon.as_arcpy, "trueCentroid", None),
                    self.spatial_reference.as_arcpy))
        elif HASARCPY:
            return Geometry(arcpy.PointGeometry(getattr(self.as_arcpy, "trueCentroid", None),
                                                self.spatial_reference.as_arcpy))
        elif HASSHAPELY:
            return self.centroid
        elif isinstance(self, Point):
            return self
        return
    #----------------------------------------------------------------------
    @property
    def geometry_type(self):
        """
        The geometry type: polygon, polyline, point, multipoint

        .. code-block:: python

            >>> geom = Geometry({
              "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
                          [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
                          [-97.06326,32.759]]],
              "spatialReference" : {"wkid" : 4326}
            })
            >>> geom.geometry_type
            'polygon'


        :returns: string
        """
        if isinstance(self, Envelope):
            return 'envelope'
        elif isinstance(self, Point):
            return 'point'
        elif isinstance(self, MultiPoint):
            return "multipoint"
        elif isinstance(self, Polyline):
            return "polyline"
        elif isinstance(self, Polygon):
            return "polygon"
        return
    #Functions#############################################################
    #----------------------------------------------------------------------
    def angle_distance_to(self, second_geometry, method="GEODESIC"):
        """
        Returns a tuple of angle and distance to another point using a
        measurement type.  If `ArcPy` is not installed, none is returned.

        **Requires ArcPy**

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
        HASARCPY, HASSHAPELY = self._check_geometry_engine()

        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            if isinstance(second_geometry, Envelope):
                second_geometry = second_geometry.polygon
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_arcpy
            return self.as_arcpy.angleAndDistanceTo(other_geometry=second_geometry,
                                                    method=method)
        return None
    #----------------------------------------------------------------------
    def boundary(self):
        """
        Constructs the boundary of the geometry.

        :returns: arcgis.geometry.Polyline
        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()

        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            return Geometry(self.as_arcpy.boundary())
        elif HASSHAPELY  and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            return Geometry(self.as_shapely.boundary.buffer(1).__geo_interface__)
        return None
    #----------------------------------------------------------------------
    def buffer(self, distance):
        """
        Constructs a polygon at a specified distance from the geometry.

        **Requires ArcPy**

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        distance            Required float. The buffer distance. The buffer distance is in the
                            same units as the geometry that is being buffered.
                            A negative distance can only be specified against a polygon geometry.
        ===============     ====================================================================

        :returns: arcgis.geometry.Polygon
        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            return Geometry(self.as_arcpy.buffer(distance))
        elif HASSHAPELY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            return Geometry(self.as_shapely.buffer(
                distance).__geo_interface__)
        return None
    #----------------------------------------------------------------------
    def clip(self, envelope):
        """
        Constructs the intersection of the geometry and the specified extent.
        If `ArcPy` is not installed, none is returned.

        **Requires ArcPy**


        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        envelope            required tuple. The tuple must have (XMin, YMin, XMax, YMax) each value
                            represents the lower left bound and upper right bound of the extent.
        ===============     ====================================================================

        :returns: output geometry clipped to extent

        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY:
            import arcpy
        if HASARCPY and \
           isinstance(envelope, (list,tuple)) and \
           len(envelope) == 4:
            envelope = arcpy.Extent(XMin=envelope[0],
                                    YMin=envelope[1],
                                    XMax=envelope[2],
                                    YMax=envelope[3])
            return Geometry(self.as_arcpy.clip(envelope))
        elif HASARCPY and \
             isinstance(self, (Point, Polygon, Polyline, MultiPoint)) and \
             isinstance(envelope, arcpy.Extent):
            return Geometry(self.as_arcpy.clip(envelope))
        return None
    #----------------------------------------------------------------------
    def contains(self, second_geometry, relation=None):
        """
        Indicates if the base geometry contains the comparison geometry.

        **Requires ArcPy/Shapely**

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
        HASARCPY, HASSHAPELY = self._check_geometry_engine()

        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_arcpy
            return self.as_arcpy.contains(second_geometry=second_geometry,
                                          relation=relation)
        elif HASSHAPELY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_shapely
            return self.as_shapely.contains(second_geometry)
        return None
    #----------------------------------------------------------------------
    def convex_hull(self):
        """
        Constructs the geometry that is the minimal bounding polygon such
        that all outer angles are convex.
        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            return Geometry(self.as_arcpy.convexHull())
        elif self.type.lower() == "polygon":
            from ._convexhull import convex_hull
            combine_pts = [pt for part in self['rings'] for pt in part]
            try:
                return Geometry({'rings' : [convex_hull(combine_pts)],
                                 'spatialReference' : self['spatialReference']})
            except:
                from ._convexhull import convex_hull_GS
                return Geometry({'rings' : [convex_hull_GS(combine_pts)],
                                 'spatialReference' : self['spatialReference']})
        elif self.type.lower() == "polyline":
            from ._convexhull import convex_hull
            combine_pts = [pt for part in self['paths'] for pt in part]
            try:
                return Geometry({'rings' : [convex_hull(combine_pts)],
                                 'spatialReference' : self['spatialReference']})
            except:
                from ._convexhull import convex_hull_GS
                return Geometry({'rings' : [convex_hull_GS(combine_pts)],
                                 'spatialReference' : self['spatialReference']})
        elif self.type.lower() == "multipoint":
            from ._convexhull import convex_hull
            combine_pts = self['points']
            try:
                return Geometry({'rings' : [convex_hull(combine_pts)],
                                 'spatialReference' : self['spatialReference']})
            except:
                from ._convexhull import convex_hull_GS
                return Geometry({'rings' : [convex_hull_GS(combine_pts)],
                                 'spatialReference' : self['spatialReference']})
        return None
    #----------------------------------------------------------------------
    def crosses(self, second_geometry):
        """
        Indicates if the two geometries intersect in a geometry of a lesser
        shape type.

        **Requires ArcPy/Shapely**

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :returns: boolean

        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            if isinstance(second_geometry, Envelope):
                second_geometry = second_geometry.polygon
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_arcpy
            return self.as_arcpy.crosses(second_geometry=second_geometry)
        elif HASSHAPELY:
            return self.as_shapely.crosses(other=second_geometry.as_shapely)
        return None
    #----------------------------------------------------------------------
    def cut(self, cutter):
        """
        Splits this geometry into a part left of the cutting polyline, and
        a part right of it.

        **Requires ArcPy**

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        cutter              Required Polyline. The cuttin polyline geometry
        ===============     ====================================================================

        :returns: a list of two geometries

        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if isinstance(cutter, Polyline) and HASARCPY:
            if isinstance(cutter, Geometry):
                cutter = cutter.as_arcpy
            return Geometry(self.as_arcpy.cut(other=cutter))
        return None
    #----------------------------------------------------------------------
    def densify(self, method, distance, deviation):
        """
        Creates a new geometry with added vertices

        **Requires ArcPy**

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
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            return Geometry(self.as_arcpy.densify(method=method,
                                                  distance=distance,
                                                  deviation=deviation))
        return None
    #----------------------------------------------------------------------
    def difference(self, second_geometry):
        """
        Constructs the geometry that is composed only of the region unique
        to the base geometry but not part of the other geometry. The
        following illustration shows the results when the red polygon is the
        source geometry.

        **Requires ArcPy/Shapely**

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :returns: arcgis.geometry.Geometry

        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_arcpy
            g = self.as_arcpy.difference(other=second_geometry)
            return Geometry(g)
        elif HASSHAPELY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_shapely
            return Geometry(self.as_shapely.difference(second_geometry).__geo_interface__)
        return None
    #----------------------------------------------------------------------
    def disjoint(self, second_geometry):
        """
        Indicates if the base and comparison geometries share no points in
        common.

        **Requires ArcPy/Shapely**

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :returns: boolean

        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY:
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_arcpy
            return self.as_arcpy.disjoint(second_geometry=second_geometry)
        elif HASSHAPELY:
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_shapely
            return self.as_shapely.disjoint(second_geometry)
        return None
    #----------------------------------------------------------------------
    def distance_to(self, second_geometry):
        """
        Returns the minimum distance between two geometries. If the
        geometries intersect, the minimum distance is 0.
        Both geometries must have the same projection.

        **Requires ArcPy/Shapely**

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :returns: float

        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            if isinstance(second_geometry, Envelope):
                second_geometry = second_geometry.polygon
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_arcpy
            return self.as_arcpy.distanceTo(other=second_geometry)
        elif HASSHAPELY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_shapely
            return self.as_shapely.distance(other=second_geometry)
        return None
    #----------------------------------------------------------------------
    def equals(self, second_geometry):
        """
        Indicates if the base and comparison geometries are of the same
        shape type and define the same set of points in the plane. This is
        a 2D comparison only; M and Z values are ignored.

        **Requires ArcPy or Shapely**

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :returns: boolean


        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY:
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_arcpy
            return self.as_arcpy.equals(second_geometry=second_geometry)
        elif HASSHAPELY:
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_shapely
            return self.as_shapely.equals(other=second_geometry)
        return None
    #----------------------------------------------------------------------
    def generalize(self, max_offset):
        """
        Creates a new simplified geometry using a specified maximum offset
        tolerance.

        **Requires ArcPy or Shapely**

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        max_offset          Required float. The maximum offset tolerance.
        ===============     ====================================================================

        :returns: arcgis.geometry.Geometry

        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            return Geometry(self.as_arcpy.generalize(distance=max_offset))
        elif HASSHAPELY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            return Geometry(self.as_shapely.simplify(
                max_offset).__geo_interface__)
        return None
    #----------------------------------------------------------------------
    def get_area(self, method, units=None):
        """
        Returns the area of the feature using a measurement type.

        **Requires ArcPy or Shapely**

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        method              Required String. LANAR measurements reflect the projection of
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
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            return self.as_arcpy.getArea(method=method,
                                         units=units)
        elif HASARCPY and isinstance(self, Envelope):
            return self.polygon.as_arcpy.getArea(method=method, units=units)
        return None
    #----------------------------------------------------------------------
    def get_length(self, method, units):
        """
        Returns the length of the feature using a measurement type.

        **Requires ArcPy or Shapely**

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
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            return self.as_arcpy.getLength(method=method,
                                           units=units)
        elif HASARCPY and isinstance(self, Envelope):
            return self.polygon.as_arcpy.getLength(method=method, units=units)
        return None
    #----------------------------------------------------------------------
    def get_part(self, index=None):
        """
        Returns an array of point objects for a particular part of geometry
        or an array containing a number of arrays, one for each part.

        **Requires ArcPy**

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        index               Required Integer. The index position of the geometry.
        ===============     ====================================================================

        :return: arcgis.geometry.Geometry

        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            return self.as_arcpy.getPart(index)
        return None
    #----------------------------------------------------------------------
    def intersect(self, second_geometry, dimension=1):
        """
        Constructs a geometry that is the geometric intersection of the two
        input geometries. Different dimension values can be used to create
        different shape types. The intersection of two geometries of the
        same shape type is a geometry containing only the regions of overlap
        between the original geometries.

        **Requires ArcPy or Shapely**

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
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            if isinstance(second_geometry, Envelope):
                second_geometry = second_geometry.polygon
                dimension = 4
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_arcpy
            return Geometry(self.as_arcpy.intersect(other=second_geometry,
                                                    dimension=dimension))
        elif HASARCPY and \
             isinstance(self, Envelope):
            if isinstance(second_geometry, Envelope):
                second_geometry = second_geometry.polygon
                dimension = 4
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_arcpy
            return Geometry(self.polygon.as_arcpy.intersect(other=second_geometry,
                                                            dimension=dimension))
        elif HASSHAPELY:
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_shapely
            return Geometry(self.as_shapely.intersection(
                other=second_geometry).__geo_interface__)
        return None
    #----------------------------------------------------------------------
    def measure_on_line(self, second_geometry, as_percentage=False):
        """
        Returns a measure from the start point of this line to the in_point.

        **Requires ArcPy**

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
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_arcpy
            return self.as_arcpy.measureOnLine(in_point=second_geometry,
                                               use_percentage=as_percentage)
        return None
    #----------------------------------------------------------------------
    def overlaps(self, second_geometry):
        """
        Indicates if the intersection of the two geometries has the same
        shape type as one of the input geometries and is not equivalent to
        either of the input geometries.

        **Requires ArcPy or Shapely**

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :return: boolean

        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY:
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_arcpy
            return self.as_arcpy.overlaps(second_geometry=second_geometry)
        elif HASSHAPELY:
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_shapely
            return self.as_shapely.overlaps(other=second_geometry)
        return None
    #----------------------------------------------------------------------
    def point_from_angle_and_distance(self, angle, distance, method='GEODESCIC'):
        """
        Returns a point at a given angle and distance in degrees and meters
        using the specified measurement type.

        **Requires ArcPy**

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
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            return Geometry(self.as_arcpy.pointFromAngleAndDistance(angle=angle,
                                                                    distance=distance,
                                                                    method=method))
        return None
    #----------------------------------------------------------------------
    def position_along_line(self, value, use_percentage=False):
        """
        Returns a point on a line at a specified distance from the beginning
        of the line.

        **Requires ArcPy or Shapely**

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

        :return: arcgis.gis.Geometry

        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            return Geometry(self.as_arcpy.positionAlongLine(value=value,
                                                            use_percentage=use_percentage))
        elif HASSHAPELY:
            return Geometry(self.as_shapely.interpolate(value, normalized=use_percentage).__geo_interface__)

        return None
    #----------------------------------------------------------------------
    def project_as(self, spatial_reference, transformation_name=None):
        """
        Projects a geometry and optionally applies a geotransformation.

        **Requires ArcPy or pyproj>=1.9 and PROJ.4**

        ====================     ====================================================================
        **Argument**             **Description**
        --------------------     --------------------------------------------------------------------
        spatial_reference        Required SpatialReference. The new spatial reference. This can be a
                                 SpatialReference object or the coordinate system name.
        --------------------     --------------------------------------------------------------------
        transformation_name      Required String. The geotransformation name.
        ====================     ====================================================================

        Parameter:
         :spatial_reference: - The new spatial reference. This can be a
          SpatialReference object or the coordinate system name.
         :transformation_name: - The geotransformation name.
        """
        from six import string_types, integer_types
        HASARCPY, HASSHAPELY = self._check_geometry_engine()

        if HASARCPY:
            import arcpy

        if HASARCPY:
            if isinstance(spatial_reference, SpatialReference):
                spatial_reference = spatial_reference.as_arcpy
            elif isinstance(spatial_reference, dict):
                spatial_reference = SpatialReference(spatial_reference).as_arcpy
            elif isinstance(spatial_reference, arcpy.SpatialReference):
                spatial_reference = spatial_reference
            elif isinstance(spatial_reference, integer_types):
                spatial_reference = arcpy.SpatialReference(spatial_reference)
            elif isinstance(spatial_reference, string_types):
                spatial_reference = arcpy.SpatialReference(
                    text=spatial_reference)
            else:
                raise ValueError("Invalid spatial reference object.")
            return Geometry(self.as_arcpy.projectAs(spatial_reference=spatial_reference,
                                                    transformation_name=transformation_name))

        try:
            import pyproj
            from shapely.ops import transform
            HASPROJ = True
        except:
            HASPROJ = False

        # Project using Proj4 (pyproj)
        if HASPROJ:

            esri_projections = {
                102100: 3857,
                102113: 3857
            }

            # Get the input spatial reference
            in_srid = self.spatial_reference.get('wkid',None)
            in_srid = self.spatial_reference.get('latestWkid',in_srid)
            # Convert web mercator from esri SRID
            in_srid = esri_projections.get(int(in_srid),in_srid)
            in_srid = 'epsg:{}'.format(in_srid)

            if isinstance(spatial_reference, dict) or isinstance(spatial_reference, SpatialReference):
                out_srid = spatial_reference.get('wkid',None)
                out_srid = spatial_reference.get('latestWkid',out_srid)
            elif isinstance(spatial_reference, integer_types):
                out_srid = spatial_reference
            elif isinstance(spatial_reference, string_types):
                out_srid = spatial_reference
            else:
                raise ValueError("Invalid spatial reference object.")

            out_srid = esri_projections.get(int(out_srid),out_srid)
            out_srid = 'epsg:{}'.format(out_srid)

            try:
                project = partial(
                    pyproj.transform,
                    pyproj.Proj(init=in_srid),
                    pyproj.Proj(init=out_srid)
                )
            except RuntimeError as e:
                raise ValueError("pyproj projection from {0} to {1} not currently supported".format(in_srid,out_srid))

            g = transform(project,self.as_shapely)
            return Geometry.from_shapely(
                g,
                spatial_reference=spatial_reference
            )

        return None
    #----------------------------------------------------------------------
    def query_point_and_distance(self, second_geometry,
                                 use_percentage=False):
        """
        Finds the point on the polyline nearest to the in_point and the
        distance between those points. Also returns information about the
        side of the line the in_point is on as well as the distance along
        the line where the nearest point occurs.

        **Requires ArcPy**

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
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_arcpy
            return self.as_arcpy.queryPointAndDistance(in_point=second_geometry,
                                                       use_percentage=use_percentage)
        return None
    #----------------------------------------------------------------------
    def segment_along_line(self, start_measure,
                           end_measure, use_percentage=False):
        """
        Returns a Polyline between start and end measures. Similar to
        Polyline.positionAlongLine but will return a polyline segment between
        two points on the polyline instead of a single point.

        **Requires ArcPy**

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

        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            return Geometry(self.as_arcpy.segmentAlongLine(
                start_measure=start_measure,
                end_measure=end_measure,
                use_percentage=use_percentage))
        return None
    #----------------------------------------------------------------------
    def snap_to_line(self, second_geometry):
        """
        Returns a new point based on in_point snapped to this geometry.

        **Requires ArcPy**

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :return: arcgis.gis.Geometry

        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_arcpy
            return Geometry(self.as_arcpy.snapToLine(in_point=second_geometry))
        return None
    #----------------------------------------------------------------------
    def symmetric_difference (self, second_geometry):
        """
        Constructs the geometry that is the union of two geometries minus the
        instersection of those geometries.

        The two input geometries must be the same shape type.

        **Requires ArcPy or Shapely**

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :return: arcgis.gis.Geometry
        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()

        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            if isinstance(second_geometry, Envelope):
                second_geometry = second_geometry.polygon
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_arcpy
            return Geometry(self.as_arcpy.symmetricDifference(other=second_geometry))
        elif HASSHAPELY:
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_shapely
            return Geometry(self.as_shapely.symmetric_difference(
                other=second_geometry).__geo_interface__)
        return None
    #----------------------------------------------------------------------
    def touches(self, second_geometry):
        """
        Indicates if the boundaries of the geometries intersect.

        **Requires ArcPy or Shapely**

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :return: boolean
        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY:
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_arcpy
            return self.as_arcpy.touches(second_geometry=second_geometry)
        elif HASSHAPELY:
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_shapely
            return self.as_shapely.touches(second_geometry)
        return None
    #----------------------------------------------------------------------
    def union(self, second_geometry):
        """
        Constructs the geometry that is the set-theoretic union of the input
        geometries.

        **Requires ArcPy or Shapely**

        ===============     ====================================================================
        **Argument**        **Description**
        ---------------     --------------------------------------------------------------------
        second_geometry     Required arcgis.geometry.Geometry. A second geometry
        ===============     ====================================================================

        :return: arcgis.gis.Geometry
        """
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            if isinstance(second_geometry, Envelope):
                second_geometry = second_geometry.polygon
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_arcpy
            return Geometry(self.as_arcpy.union(other=second_geometry))
        elif HASSHAPELY and isinstance(self, (Point, Polygon, Polyline, MultiPoint)):
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_shapely
            return Geometry(self.as_shapely.union(
                second_geometry).__geo_interface__)
        return None
    #----------------------------------------------------------------------
    def within(self, second_geometry, relation=None):
        """
        Indicates if the base geometry is within the comparison geometry.

        **Requires ArcPy or Shapely**

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
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY:
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_arcpy
            return self.as_arcpy.within(second_geometry=second_geometry,
                                        relation=relation)
        elif HASSHAPELY:
            if isinstance(second_geometry, Geometry):
                second_geometry = second_geometry.as_shapely
            return self.as_shapely.within(second_geometry)
        return None
###########################################################################
class MultiPoint(Geometry):
    """
    A multipoint contains an array of points, along with a spatialReference
    field. A multipoint can also have boolean-valued hasZ and hasM fields.
    These fields control the interpretation of elements of the points
    array. Omitting an hasZ or hasM field is equivalent to setting it to
    false.
    Each element of the points array is itself an array of two, three, or
    four numbers. It will have two elements for 2D points, two or three
    elements for 2D points with Ms, three elements for 3D points, and three
    or four elements for 3D points with Ms. In all cases, the x coordinate
    is at index 0 of a point's array, and the y coordinate is at index 1.
    For 2D points with Ms, the m coordinate, if present, is at index 2. For
    3D points, the Z coordinate is required and is at index 2. For 3D
    points with Ms, the Z coordinate is at index 2, and the M coordinate,
    if present, is at index 3.
    An empty multipoint has a points field with no elements. Empty points
    are ignored.
    """
    _type = "Multipoint"
    def __init__(self, iterable=None,
                 **kwargs):
        if iterable is None:
            iterable = ()
        super(MultiPoint, self).__init__(iterable)
        self.update(kwargs)
    @property
    def __geo_interface__(self):
        return {'type': 'Multipoint', 'coordinates': [(pt[0], pt[1]) for pt in self['points']]}
    #----------------------------------------------------------------------
    @property
    def type(self):
        return self._type
    #----------------------------------------------------------------------
    def svg(self, scale_factor=1., fill_color=None):
        """Returns a group of SVG circle elements for the MultiPoint geometry.

        Parameters
        ==========
        scale_factor : float
            Multiplication factor for the SVG circle diameters.  Default is 1.
        fill_color : str, optional
            Hex string for fill color. Default is to use "#66cc99" if
            geometry is valid, and "#ff3333" if invalid.
        """
        if self.is_empty:
            return '<g />'
        if fill_color is None:
            fill_color = "#66cc99" if self.is_valid else "#ff3333"
        return '<g>' + \
               ''.join(('<circle cx="{0.x}" cy="{0.y}" r="{1}" '
                        'stroke="#555555" stroke-width="{2}" fill="{3}" opacity="0.6" />'
                        ).format(Point({'x': p[0], 'y': p[1]}), 3 * scale_factor, 1 * scale_factor, fill_color) \
                       for p in self['points']) + \
               '</g>'
    #----------------------------------------------------------------------
    def __hash__(self):
        return hash(json.dumps(dict(self)))
    #----------------------------------------------------------------------
    def coordinates(self):
        """returns the coordinates as a np.array"""
        import numpy as np
        if 'points' in self:
            return np.array(self['points'])
        else:
            return np.array([])
    #----------------------------------------------------------------------
    def __setstate__(self, d):
        """unpickle support """
        self.__dict__.update(d)
        self = MultiPoint(iterable=d)
    #----------------------------------------------------------------------
    def __getstate__(self):
        """ pickle support """
        return dict(self)
    #----------------------------------------------------------------------
    @classmethod
    def _from_geojson(cls, data, sr=None):
        if sr is None:
            sr = {'wkid' : 4326}

        coordkey = 'coordinates'
        for d in data:
            if d.lower() == 'coordinates':
                coordkey = d

        coordinates = data[coordkey]

        return cls({'points' : [p for p in coordinates],
                    'spatialReference' : sr})
########################################################################
class Point(Geometry):
    """
    A point contains x and y fields along with a spatialReference field. A
    point can also contain m and z fields. A point is empty when its x
    field is present and has the value null or the string "NaN". An empty
    point has no location in space.
    """
    _type = "Point"
    #----------------------------------------------------------------------
    def __init__(self, iterable=None):
        """Constructor"""
        super(Point, self)
        if iterable is None:
            iterable = {}
        self.update(iterable)
    #----------------------------------------------------------------------
    @property
    def type(self):
        return self._type
    #----------------------------------------------------------------------
    def svg(self, scale_factor=1, fill_color=None):
        """Returns SVG circle element for the Point geometry.

        Parameters
        ==========
        scale_factor : float
            Multiplication factor for the SVG circle diameter.  Default is 1.
        fill_color : str, optional
            Hex string for fill color. Default is to use "#66cc99" if
            geometry is valid, and "#ff3333" if invalid.
        """
        if self.is_empty:
            return '<g />'
        if fill_color is None:
            fill_color = "#66cc99" if self.is_valid else "#ff3333"
        return (
            '<circle cx="{0.x}" cy="{0.y}" r="{1}" '
            'stroke="#555555" stroke-width="{2}" fill="{3}" opacity="0.6" />'
            ).format(self, 3 * scale_factor, 1 * scale_factor, fill_color)
    #----------------------------------------------------------------------
    def __setstate__(self, d):
        """unpickle support """
        self.__dict__.update(d)
        self = Point(iterable=d)
    #----------------------------------------------------------------------
    def __getstate__(self):
        """ pickle support """
        return dict(self)
    #----------------------------------------------------------------------
    def __hash__(self):
        return hash(json.dumps(dict(self)))
    #----------------------------------------------------------------------
    def coordinates(self):
        """returns the coordinates as a np.array"""
        import numpy as np
        if 'x' in self and 'y' in self and 'z' in self:
            return np.array([self['x'], self['y'], self['z']])
        elif 'x' in self and 'y' in self:
            return np.array([self['x'], self['y']])
        else:
            return np.array([])
    @classmethod
    def _from_geojson(cls, data, sr=None):
        if sr == None:
            sr = {'wkid' : 4326}

        coordkey = 'coordinates'
        for d in data:
            if d.lower() == 'coordinates':
                coordkey = d
        coordinates = data[coordkey]

        return cls({
            "x" : coordinates[0],
            "y" : coordinates[1],
            "spatialReference" : sr
        })

########################################################################
class Polygon(Geometry):
    """
    A polygon contains an array of rings or curveRings and a
    spatialReference. For polygons with curveRings, see the sections on
    JSON curve object and Polygon with curve. Each ring is represented as
    an array of points. The first point of each ring is always the same as
    the last point. Each point in the ring is represented as an array of
    numbers. A polygon can also have boolean-valued hasM and hasZ fields.

    An empty polygon is represented with an empty array for the rings
    field. Nulls and/or NaNs embedded in an otherwise defined coordinate
    stream for polylines/polygons is a syntax error.
    Polygons should be topologically simple. Exterior rings are oriented
    clockwise, while holes are oriented counter-clockwise. Rings can touch
    at a vertex or self-touch at a vertex, but there should be no other
    intersections. Polygons returned by services are topologically simple.
    When drawing a polygon, use the even-odd fill rule. The even-odd fill
    rule will guarantee that the polygon will draw correctly even if the
    ring orientation is not as described above.
    """
    _type = "Polygon"
    def __init__(self, iterable=None,
                 **kwargs):
        if iterable is None:
            iterable = ()
        super(Polygon, self).__init__(iterable)
        self.update(kwargs)
    #----------------------------------------------------------------------
    def svg(self, scale_factor=1,fill_color=None):
        if self.is_empty:
            return '<g />'
        if fill_color is None:
            fill_color = "#66cc99" if self.is_valid else "#ff3333"
        rings = []
        s = ""

        if 'rings' not in self:
            densify_geom = self.densify('ANGLE', -1, 0.1)
            geom_json = json.loads(densify_geom.JSON)['rings']
        else:
            geom_json = self['rings']
        for ring in geom_json:
            rings = ring
            exterior_coords = [
                ["{},{}".format(*c) for c in rings]]
            path = " ".join([
                "M {} L {} z".format(coords[0], " L ".join(coords[1:]))
                for coords in exterior_coords])
            s += (
            '<path fill-rule="evenodd" fill="{2}" stroke="#555555" '
            'stroke-width="{0}" opacity="0.6" d="{1}" />'
            ).format(2. * scale_factor, path, fill_color)
        return s
    #----------------------------------------------------------------------
    @property
    def type(self):
        return self._type
    #----------------------------------------------------------------------
    def __hash__(self):
        return hash(json.dumps(dict(self)))
    #----------------------------------------------------------------------
    def coordinates(self):
        """returns the coordinates as a np.array"""
        import numpy as np
        if 'rings' in self:
            return np.array(self['rings'])
        else:
            return np.array([])
    #----------------------------------------------------------------------
    def __setstate__(self, d):
        """unpickle support """
        self.__dict__.update(d)
        self = Polygon(iterable=d)
    #----------------------------------------------------------------------
    def __getstate__(self):
        """ pickle support """
        return dict(self)
    @classmethod
    def _from_geojson(cls, data, sr=None):
        if sr is None:
            sr = {'wkid' : 4326}

        coordkey = 'coordinates'
        for d in data:
            if d.lower() == 'coordinates':
                coordkey = d
        coordinates = data[coordkey]
        typekey = 'type'
        for d in data:
            if d.lower() == 'type':
                typekey = d

        if data[typekey].lower() == "polygon":
            coordinates = [coordinates]
        part_list = []
        for part in coordinates:
            part_item = []
            for ring in part:
                for coord in reversed(ring):
                    part_item.append(coord)
            if part_item:
                part_list.append(part_item)
        return cls({'rings' : part_list,
                    'spatialReference' : sr
                    })
########################################################################
class Polyline(Geometry):
    """
    A polyline contains an array of paths or curvePaths and a
    spatialReference. For polylines with curvePaths, see the sections on
    JSON curve object and Polyline with curve. Each path is represented as
    an array of points, and each point in the path is represented as an
    array of numbers. A polyline can also have boolean-valued hasM and hasZ
    fields.
    See the description of multipoints for details on how the point arrays
    are interpreted.
    An empty polyline is represented with an empty array for the paths
    field. Nulls and/or NaNs embedded in an otherwise defined coordinate
    stream for polylines/polygons is a syntax error.
    """
    _type = "Polyline"
    def __init__(self, iterable=None,
                 **kwargs):
        if iterable is None:
            iterable = {}
        super(Polyline, self).__init__(iterable)
        self.update(kwargs)
    #----------------------------------------------------------------------
    def svg(self, scale_factor=1, stroke_color=None):
        """Returns SVG polyline element for the LineString geometry.

        Parameters
        ==========
        scale_factor : float
            Multiplication factor for the SVG stroke-width.  Default is 1.
        stroke_color : str, optional
            Hex string for stroke color. Default is to use "#66cc99" if
            geometry is valid, and "#ff3333" if invalid.
        """
        if self.is_empty:
            return '<g />'
        if stroke_color is None:
            stroke_color = "#66cc99" if self.is_valid else "#ff3333"
        paths = []

        if 'paths' not in self:
            densify_geom = self.densify('DISTANCE', 1.0, 0.1)
            geom_json = json.loads(densify_geom.JSON)['paths']
        else:
            geom_json = self['paths']
        for path in geom_json:
            pnt_format = " ".join(["{0},{1}".format(*c) for c in path])
            s = ('<polyline fill="none" stroke="{2}" stroke-width="{1}" '
                 'points="{0}" opacity="0.8" />').format(pnt_format, 2. * scale_factor, stroke_color)
            paths.append(s)
        return "<g>" + "".join(paths) + "</g>"
    #----------------------------------------------------------------------
    @property
    def type(self):
        return self._type
    #----------------------------------------------------------------------
    def __hash__(self):
        return hash(json.dumps(dict(self)))
    #----------------------------------------------------------------------
    def coordinates(self):
        """returns the coordinates as a np.array"""
        import numpy as np
        if 'paths' in self:
            return np.array(self['paths'])
        else:
            return np.array([])
    #----------------------------------------------------------------------
    @property
    def __geo_interface__(self):
        return {'type': 'MultiLineString', 'coordinates': [[((pt[0], pt[1]) if pt else None)
                                                            for pt in part]
                                                           for part in self['paths']]}
    #----------------------------------------------------------------------
    def __setstate__(self, d):
        """unpickle support """
        self.__dict__.update(d)
        self = Polyline(iterable=d)
    #----------------------------------------------------------------------
    def __getstate__(self):
        """ pickle support """
        return dict(self)
    @classmethod
    def _from_geojson(cls, data, sr=None):
        if sr is None:
            sr = {'wkid' : 4326}
        if data['type'].lower() == 'linestring':
            coordinates = [data['coordinates']]
        else:
            coordinates = data['coordinates']

        return cls(
            {'paths' : [[p for p in part] for part in coordinates],
             'spatialReference' : sr
             })
########################################################################
class Envelope(Geometry):
    """
    An envelope is a rectangle defined by a range of values for each
    coordinate and attribute. It also has a spatialReference field. The
    fields for the z and m ranges are optional. An empty envelope has no
    in space and is defined by the presence of an xmin field a null value
    or a "NaN" string.
    """
    _type = "Envelope"
    def __init__(self, iterable=None, **kwargs):
        if iterable is None:
            iterable = ()
        super(Envelope, self).__init__(iterable)
        self.update(kwargs)
    #----------------------------------------------------------------------
    @property
    def type(self):
        return self._type
    #----------------------------------------------------------------------
    def __hash__(self):
        return hash(json.dumps(dict(self)))
    #----------------------------------------------------------------------
    def svg(self, scale_factor=1, fill_color=None):
        """"""
        return self.polygon.svg(scale_factor, fill_color)
    #----------------------------------------------------------------------
    def _repr_svg_(self):
        """SVG representation for iPython notebook"""
        return self.polygon._repr_svg_()
    #----------------------------------------------------------------------
    def coordinates(self):
        """returns the coordinates as a np.array"""
        import numpy as np
        if 'xmin' in self and \
           'xmax' in self and \
           'ymin' in self and \
           'ymax' in self:
            if 'zmin' in self and 'zmax' in self:
                return np.array([self['xmin'], self['ymin'], self['zmin'],
                                 self['xmax'], self['ymax'], self['zmax']])
            return np.array([self['xmin'], self['ymin'], self['xmax'], self['ymax']])
        else:
            return np.array([])
    #----------------------------------------------------------------------
    @property
    def geohash(self):
        """A geohash string of the extent is returned."""
        return getattr(self.as_arcpy, 'geohash', None)
    #----------------------------------------------------------------------
    @property
    def geohash_covers(self):
        """Returns a list of up to the four longest geohash strings that fit within the extent."""
        return getattr(self.as_arcpy, 'geohashCovers', None)
    #----------------------------------------------------------------------
    @property
    def geohash_neighbors(self):
        """A list of the geohash neighbor strings for the extent is returned."""
        return getattr(self.as_arcpy, 'geohashNeighbors', None)
    #----------------------------------------------------------------------
    @property
    def height(self):
        """The extent height value."""
        return getattr(self.as_arcpy, 'height', None)
    #----------------------------------------------------------------------
    @property
    def width(self):
        """The extent width value."""
        return getattr(self.as_arcpy, 'width', None)
    #----------------------------------------------------------------------
    @property
    def polygon(self):
        """returns the envelope as a Polygon geometry"""
        fe = self.coordinates().tolist()
        if 'spatialReference' in self:
            sr = SpatialReference(self['spatialReference'])
        else:
            sr = SpatialReference({'wkid' : 4326})
        return Geometry(
            {'rings' : [[[fe[0], fe[1]], [fe[0],fe[3]], [fe[2],fe[3]], [fe[2],fe[1]], [fe[0],fe[1]]]],
             'spatialReference' : sr})

    #----------------------------------------------------------------------
    def __setstate__(self, d):
        """unpickle support """
        self.__dict__.update(d)
        self = Envelope(iterable=d)
    #----------------------------------------------------------------------
    def __getstate__(self):
        """ pickle support """
        return dict(self)
########################################################################
class SpatialReference(BaseGeometry):
    """
    A spatial reference can be defined using a well-known ID (wkid) or
    well-known text (wkt). The default tolerance and resolution values for
    the associated coordinate system are used. The xy and z tolerance
    values are 1 mm or the equivalent in the unit of the coordinate system.
    If the coordinate system uses feet, the tolerance is 0.00328083333 ft.
    The resolution values are 10x smaller or 1/10 the tolerance values.
    Thus, 0.0001 m or 0.0003280833333 ft. For geographic coordinate systems
    using degrees, the equivalent of a mm at the equator is used.
    The well-known ID (WKID) for a given spatial reference can occasionally
    change. For example, the WGS 1984 Web Mercator (Auxiliary Sphere)
    projection was originally assigned WKID 102100, but was later changed
    to 3857. To ensure backward compatibility with older spatial data
    servers, the JSON wkid property will always be the value that was
    originally assigned to an SR when it was created.
    An additional property, latestWkid, identifies the current WKID value
    (as of a given software release) associated with the same spatial
    reference.
    A spatial reference can optionally include a definition for a vertical
    coordinate system (VCS), which is used to interpret the z-values of a
    geometry. A VCS defines units of measure, the location of z = 0, and
    whether the positive vertical direction is up or down. When a vertical
    coordinate system is specified with a WKID, the same caveat as
    mentioned above applies. There are two VCS WKID properties: vcsWkid and
    latestVcsWkid. A VCS WKT can also be embedded in the string value of
    the wkt property. In other words, the WKT syntax can be used to define
    an SR with both horizontal and vertical components in one string. If
    either part of an SR is custom, the entire SR will be serialized with
    only the wkt property.
    Starting at 10.3, Image Service supports image coordinate systems.
    """
    _type = "SpatialReference"
    def __init__(self,
                 iterable=None,
                 **kwargs):
        super(SpatialReference, self)
        if iterable is None:
            iterable = {}
        if isinstance(iterable, int):
            iterable = {'wkid' : iterable}
        if isinstance(iterable, str):
            iterable = {'wkt' : iterable}
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY:
            import arcpy
        if HASARCPY and \
           isinstance(iterable, arcpy.SpatialReference):
            if iterable.factoryCode:
                iterable = {'wkid' : iterable.factoryCode}
            else:
                iterable = {'wkt' : iterable.exportToString()}
        if len(iterable) > 0:
            self.update(iterable)
        if len(kwargs) > 0:
            self.update(kwargs)
    #----------------------------------------------------------------------
    @property
    def type(self):
        return self._type
    #----------------------------------------------------------------------
    def __hash__(self):
        return hash(json.dumps(dict(self)))
    #----------------------------------------------------------------------
    _repr_svg_ = None
    #----------------------------------------------------------------------
    def svg(self, scale_factor=1, fill_color=None):
        """represents the SVG of the object"""
        return "<g/>"
    #----------------------------------------------------------------------
    def __eq__(self, other):
        """checks if the spatial reference is not equal"""
        if 'wkt' in self and \
           'wkt' in other and \
           self['wkt'] == other['wkt']:
            return True
        elif 'wkid' in self and \
           'wkid' in other and \
           self['wkid'] == other['wkid']:
            return True
        return False
    #----------------------------------------------------------------------
    def __ne__(self, other):
        """checks if the two values are unequal"""
        return self.__eq__(other) == False
    #----------------------------------------------------------------------
    @property
    def as_arcpy(self):
        """returns the class as an arcpy SpatialReference object"""
        HASARCPY, HASSHAPELY = self._check_geometry_engine()
        if HASARCPY:
            import arcpy
            if 'wkid' in self:
                return arcpy.SpatialReference(self['wkid'])
            elif 'wkt' in self:
                sr = arcpy.SpatialReference()
                sr.loadFromString(self['wkt'])
                return sr
        return None
    #----------------------------------------------------------------------
    def __setstate__(self, d):
        """unpickle support """
        self.__dict__.update(d)
        self = SpatialReference(iterable=d)
    #----------------------------------------------------------------------
    def __getstate__(self):
        """ pickle support """
        return dict(self)

