"""
The arcgis.geometry module defines useful geometry types for working with geographic information and GIS functionality.
It provides functions which use geometric types as input and output as well as functions for easily converting
geometries between different representations.

Several functions accept geometries represented as dictionaries and the geometry objects in this module behave like them
as well as support the '.' (dot) notation providing attribute access.

Example:

.. code-block:: python

    >>> pt = Point({"x" : -118.15, "y" : 33.80, "spatialReference" : {"wkid" : 4326}})
    >>> print (pt.is_valid)
    True
    >>> print (pt.type) # POINT
    'POINT'
    >>> print (pt)
    '{"x" : -118.15, "y" : 33.80, "spatialReference" : {"wkid" : 4326}}'
    >>> print (pt.x, pt.y)
    (-118.15,33.80)

Example Polyline:

.. code-block:: python


    >>> line = {
      "paths" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832]],
                 [[-97.06326,32.759],[-97.06298,32.755]]],
      "spatialReference" : {"wkid" : 4326}
    }
    >>> polyline = Polyline(line)
    >>> print(polyline)
    '{"paths" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832]],[[-97.06326,32.759],[-97.06298,32.755]]],"spatialReference" : {"wkid" : 4326}}'
    >>> print(polyline.is_valid)
    True

Example of invalid geometry:

.. code-block:: python

    >>> line = {
      "paths" : [[[-97.06138],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832]],
                 [[-97.06326,32.759],[-97.06298,32.755]]],
      "spatialReference" : {"wkid" : 4326}
    }
    >>> polyline = Polyline(line)
    >>> print(polyline)
    '''{"paths" : [[[-97.06138],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832]],
    [[-97.06326,32.759],[-97.06298,32.755]]],"spatialReference" : {"wkid" : 4326}}'''
    >>>print(polyline.is_valid)
    False

The same pattern can be used repeated for Polygon, MultiPoint and SpatialReference.

You can create a Geometry even when you don't know the exact type. The Geometry constructor is able to figure out the
geometry type and returns the correct type as the example below demonstrates:

.. code-block:: python

    >>> geom = Geometry({
      "rings" : [[[-97.06138,32.837],[-97.06133,32.836],[-97.06124,32.834],[-97.06127,32.832],
                  [-97.06138,32.837]],[[-97.06326,32.759],[-97.06298,32.755],[-97.06153,32.749],
                  [-97.06326,32.759]]],
      "spatialReference" : {"wkid" : 4326}
    })
    >>> print (geom.type) # Polygon
    'Polygon'
    >>> print(isinstance(geom, Polygon)
    True

"""

from ._types import *
from .functions import *
from . import filters
