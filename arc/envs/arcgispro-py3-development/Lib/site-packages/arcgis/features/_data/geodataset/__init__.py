"""
spatial dataframe is a collection of custom dataframes and series that will
allow users to quickly manage geo-spatial data through a panda's dataframe.
"""
from __future__ import print_function
from __future__ import division
from .geodataframe import SpatialDataFrame
from .geoseries import GeoSeries
from .tools import spatial_join
try:
    from .io import from_featureclass, to_featureclass, to_sqlite, from_layer
except:
    pass

__version__ = "0.0.25"