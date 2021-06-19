"""
Functions to filter query results by a spatial relationship with another geometry
Used when querying feature layers and imagery layers.
"""

from . import Geometry

# esriSpatialRelIntersects | esriSpatialRelContains | esriSpatialRelCrosses | esriSpatialRelEnvelopeIntersects | \
# esriSpatialRelIndexIntersects | esriSpatialRelOverlaps | esriSpatialRelTouches | esriSpatialRelWithin

def _filter(geometry, sr, rel):
    if not isinstance(geometry, Geometry):
        geometry = Geometry(geometry)

    gt = {
        "point" : "Point",
        "multipoint" : "Multipoint",
        "polygon" : "Polygon",
        "polyline" : "Polyline",
        "envelope" : "Envelope"
    }

    filter = {
        'geometry': geometry,
        'geometryType': 'esriGeometry' + gt[str(geometry.type).lower()] ,
        'spatialRel': rel,
    }

    if sr is None:
        if 'spatialReference' in geometry:
            sr = geometry['spatialReference']

    if sr is not None:
        filter['inSR'] = sr

    return filter

def intersects(geometry, sr=None):
    """filters results whose geometry intersects with the specified geometry

    .. code-block:: python

        USAGE EXAMPLE: Select the gas lines that intersect a specific
        freeway feature, United States Interstate 15.

        from arcgis.geometry import Geometry
        from arcgis.geometry.filters import intersects

        # select a filter feature to construct its geometry
        rte15_fset = freeway_lyr.query(where="ROUTE_NUM = 'I15'")
        rte15_geom_dict = rte15_fset.features[0].geometry
        rte15_geom_dict['spatialReference'] = freeway_sr
        rte15_geom = Geometry(rte15_geom_dict)

        # construct a geometry filter using the filter geometry
        flyr_filter = intersects(rte15_geom, sr=freeway_sr)

        # query a feature layer for features that meet filter criteria
        gas_lines_I15 = gas_line_lyr.query(geometry_filter=flyr_filter)

    """

    return _filter(geometry, sr, 'esriSpatialRelIntersects')

def contains(geometry, sr=None):
    """Returns a feature if its shape is wholly contained within the search geometry.
    Valid for all shape type combinations."""
    return _filter(geometry, sr, 'esriSpatialRelContains')

def crosses(geometry, sr=None):
    """Returns a feature if the intersection of the interiors of the two shapes is not empty and has a lower dimension
     than the maximum dimension of the two shapes. Two lines that share an endpoint in common do not cross.
     Valid for Line/Line, Line/Area, Multi-point/Area, and Multi-point/Line shape type combinations."""
    return _filter(geometry, sr, 'esriSpatialRelCrosses')

def envelope_intersects(geometry, sr=None):
    """Returns features if the envelope of the two shapes intersects. """
    return _filter(geometry, sr, 'esriSpatialRelEnvelopeIntersects')

def index_intersects(geometry, sr=None):
    """Returns a feature if the envelope of the query geometry intersects the index entry for the target geometry."""
    return _filter(geometry, sr, 'esriSpatialRelIndexIntersects')

def overlaps(geometry, sr=None):
    """Returns a feature if the intersection of the two shapes results in an object of the same dimension, but different
     from both of the shapes. Applies to Area/Area, Line/Line, and Multi-point/Multi-point shape type combinations."""
    return _filter(geometry, sr, 'esriSpatialRelOverlaps')

def touches(geometry, sr=None):
    """Returns a feature if the two shapes share a common boundary. However, the intersection of the interiors of the
    two shapes must be empty. In the Point/Line case, the point may touch an endpoint only of the line. Applies to all
    combinations except Point/Point."""
    return _filter(geometry, sr, 'esriSpatialRelTouches')

def within(geometry, sr=None):
    """Returns a feature if its shape wholly contains the search geometry. Valid for all shape type combinations."""
    return _filter(geometry, sr, 'esriSpatialRelWithin')
